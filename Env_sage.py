from Args import parse_args
from daisy.utils.loader import load_rate, get_ur, build_candidates_set, load_attack_data_get_attack_instances
from daisy.utils.map import neaten_id, get_unpopular, find_neighbor_items
from daisy.utils.splitter import split_test
from daisy.utils.sampler import Sampler
from daisy.utils.data import PairData, PointData
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k
import random
import torch
import os
import numpy as np
import json
import pickle
import time
from tqdm import tqdm
import pandas as pd
import dgl
import sys
import copy
sys.path.append('debias-gcn')
from models import GraphSAGE
from predictors import DotLinkPredictor
from utils import compute_entropy_loss,compute_metric
import itertools
def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
import torch
from torch.optim import Adam
from torchkge.data_structures import KnowledgeGraph
from torchkge.evaluation import LinkPredictionEvaluator,TripletClassificationEvaluator
from torchkge.models import TransEModel
from torchkge.utils import Trainer, MarginLoss

class Env():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu" )
        # print('self.device:',self.device)
        self._data_path = f'data/ds/{args.dataset}'
        self.save_path = f'{args.processed_path}_{args.dataset}'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # attack parameters
        self.episode_length = args.episode_length
        # preprocess dataset
        df, self.user_num, self.item_num = load_rate(args.dataset, args.prepro, binary=True, level=args.level)
        # clean id + KG
        df, self.item_num_in_kg, self.n_entity, self.n_relation, self.kg, self.kg_dict, self.adj_entity, self.adj_relation, self.degree_item = neaten_id(args.dataset, df, args.kg_neighbor_size)
        self.degree = list(dict(sorted(self.degree_item.items(), key=lambda item: item[1],reverse=True)).keys())
        self.large_degree_items = self.degree[:int(len(self.degree)*0.3)]
        # generate KGE
        KGFILE = {'ml-1m':'ml1m-kg1m','ml-20m':'ml20m-kg500k','bx':'bx-kg150k','lastfm':'lastfm-kg15k'}
        if not os.path.exists(f'data/kg/{KGFILE[args.dataset]}/ent_embedding.pt'):
            print('============KGE============')
            kg_df = pd.DataFrame(self.kg, columns=['from', 'rel', 'to'])
            kg_df = kg_df.reindex(columns = ['from','to','rel'])
            self.train_KGE(kg_df,f'data/kg/{KGFILE[args.dataset]}/ent_embedding.pt')
        
        # print('len of rating',len(df))
        # Unpopular + popular
        unpopular_items,self.popular_items = get_unpopular(df, self.item_num_in_kg, args.unpop_ratio)  # target item should be in KG.      
        # item pool
        self.item_pool = set(range(self.item_num))
        # target_item
        self.target_items = random.sample(unpopular_items, args.num_target_item)
        print(f'=============Env target items ===============')
        print(self.target_items)
        # adj_item_dict
        self.adj_item_dict = find_neighbor_items(f'{self._data_path}/adj_item_{args.prepro}_{args.level}_{args.kg_hop}.pkl', self.adj_entity, self.item_num_in_kg, args.kg_hop)
        
        # set gym variable
        # self.action_space = gym.spaces.Discrete(self.item_num) # actoin choose from item id [0, self.item_num-1]
        # self.observation_space = gym.spaces.MultiDiscrete([self.item_num]*self.episode_length) # len(episode) dimension, each dimension could be [0,self.item_num -1 ]
        
        # ========================= train target model =======================================
        
        # Negative sampler
        self.spler = Sampler(
            self.user_num+args.num_max_attacker, 
            self.item_num, 
            num_ng=args.num_ng, 
            sample_method=args.sample_method, 
            sample_ratio=args.sample_ratio
        )
        # initial random feature table
        self.emb_lookup = np.random.rand(self.item_num+self.user_num+args.num_max_attacker,  args.init_dim)
        node_features = torch.from_numpy(self.emb_lookup[list(range(self.item_num+self.user_num))])
        edges_src = torch.from_numpy(df['user'].to_numpy()+ self.item_num)# note: user to homograph node id
        edges_dst = torch.from_numpy(df['item'].to_numpy())
        # DGL graph
        self.graph = dgl.graph((edges_src,edges_dst), num_nodes = node_features.shape[0])
        self.graph.ndata['feat'] = node_features
        self.features = self.graph.ndata['feat'].to(self.device)
        # print(self.features)
        # print("=====total graph======")
        # print(self.graph)
        # Train, Test
        self.train_set, self.test_set = split_test(df, 'ufo', args.test_size)
        self.test_ur = get_ur(self.test_set)
        self.test_ucands, self.train_spy, self.eval_spy = build_candidates_set(self.test_ur, get_ur(self.train_set), self.item_pool, args, self.target_items)
        # print("=====train set======")
        # print(self.train_set)
        # print("=====test set======")
        # print(self.test_set)
        if not os.path.exists(self.save_path+f'/split_{args.num_ng}_sage_seed{self.args.seed}.pkl'): # the split are got under seed 28 for ml-1m
            self.train_neg = self.spler.transform(self.train_set, is_training=True)
            self.test_neg = self.spler.transform(self.test_set, is_training=True)
            split_dict = {'train_neg':self.train_neg, 'test_neg':self.test_neg}
            with open(self.save_path+f'/split_{args.num_ng}_sage_seed{self.args.seed}.pkl','wb') as f:
                pickle.dump(split_dict,f)
        else:
            with open(self.save_path+f'/split_{args.num_ng}_sage_seed{self.args.seed}.pkl','rb') as f:
                split_dict = pickle.load(f)
            self.train_neg = split_dict['train_neg']
            self.test_neg = split_dict['test_neg']
        # train positive, train negative, test positive, test negative
        train_pair = pd.DataFrame(
            self.train_neg,
            columns = ['user','item','rating','neg_set']
        )
        # print("=====train pair======")
        # print(train_pair)
        train_pair = train_pair.explode('neg_set')[['user','item', 'neg_set']]
        self.train_negative = train_pair[['user','neg_set']].reset_index(drop=True).rename(columns={'neg_set':'item'}).astype('int64')
        self.train_positive = train_pair[['user','item']].reset_index(drop=True)
        # print("=====train negative======")
        # print(self.train_negative)
        # print(self.train_negative.dtypes)
        # print("=====train positive======")
        # print(self.train_positive)
        # print(self.train_positive.dtypes)
        test_pair = pd.DataFrame(
            self.test_neg,
            columns = ['user','item','rating','neg_set']
        )
        test_pair = test_pair.explode('neg_set')[['user','item','neg_set']]
        self.test_negative = test_pair[['user','neg_set']].reset_index(drop=True).rename(columns={'neg_set':'item'}).astype('int64')
        self.test_positive = test_pair[['user','item']].reset_index(drop=True)
        # print("=====test negative======")
        # print(self.test_negative)
        # print(self.test_negative.dtypes)
        # print("=====test positive======")
        # print(self.test_positive)
        # print(self.test_positive.dtypes)
        # idx for oversampling positives
        self.test_index_dict = self.test_positive.reset_index().groupby('user',sort=False)['index'].apply(list).to_dict()
        self.train_index_dict = self.train_positive.reset_index().groupby('user',sort=False)['index'].apply(list).to_dict()
        
        test_eids = self.graph.edge_ids(torch.tensor(self.test_positive['user'].to_numpy()+self.item_num),torch.tensor(self.test_positive['item'].to_numpy()))
        # print("====test_eids=====")
        # print(test_eids.shape)
        # tranform to tensor
        test_pos_u, test_pos_v = torch.from_numpy(self.test_positive['user'].to_numpy()+self.item_num), torch.from_numpy(self.test_positive['item'].to_numpy())
        test_neg_u, test_neg_v = torch.from_numpy(self.test_negative['user'].to_numpy()+self.item_num), torch.from_numpy(self.test_negative['item'].to_numpy())
        train_pos_u, train_pos_v = torch.from_numpy(self.train_positive['user'].to_numpy()+self.item_num), torch.from_numpy(self.train_positive['item'].to_numpy())
        train_neg_u, train_neg_v = torch.from_numpy(self.train_negative['user'].to_numpy()+self.item_num), torch.from_numpy(self.train_negative['item'].to_numpy())
        

        # print ('==== Link Prediction Data ====')
        # print ('  TrainPosEdges: ', len(train_pos_u))
        # print ('  TrainNegEdges: ', len(train_neg_u))
        # print ('  TestPosEdges: ', len(test_pos_u))
        # print ('  TestNegEdges: ', len(test_neg_u))

        # Remove the edges in the test set from the original graph:
        # -  A subgraph will be created from the original graph by ``dgl.remove_edges``
        self.train_graph = dgl.remove_edges(self.graph, test_eids).to(self.device) # test_eids里面有很多重复，但是没有关系，重复的边只会删除一次
        # print("=====train graph======")
        # print(self.train_graph)
        # Construct positive graph and negative graph
        # -  Positive graph consists of all the positive examples as edges
        # -  Negative graph consists of all the negative examples
        # -  Both contain the same set of nodes as the original graph
        self.train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=self.graph.number_of_nodes()).to(self.device)
        self.train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=self.graph.number_of_nodes()).to(self.device)

        self.test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=self.graph.number_of_nodes()).to(self.device)
        self.test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=self.graph.number_of_nodes()).to(self.device)

        self._build_model()
        print('===========before attack=========')
        for i in self.target_items:
            res = self.evaluate(self.train_spy,self.train_graph, self.features, target_item = i)
            print(f'target [{i}] , [{res}]')
        
    def _build_model(self):
        # print('====features=====')
        # print(self.features)
        self.model = GraphSAGE(
                      in_dim=self.features.shape[1], 
                      hidden_dim=self.args.dim1, 
                      out_dim=self.args.dim2).to(self.device).double()
        # print('====model=====')
        # print(self.model)
        if os.path.exists(self.save_path+f'/{self.args.init_dim}_{self.args.dim1}_{self.args.dim2}_epoch{self.args.sage_epoch}.pkl'):
            # print('====load model=====')
            self.model = torch.load(self.save_path+f'/{self.args.init_dim}_{self.args.dim1}_{self.args.dim2}_epoch{self.args.sage_epoch}.pkl',map_location = 'cuda:0').to(self.device).double()
            return
        self.pred = DotLinkPredictor().to(self.device)
        
        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.pred.parameters()), lr=0.001)
        # print ('==== Training ====')
        # Training loop
        dur = []
        cur = time.time()
        for e in range(self.args.sage_epoch):
            self.model.train()
            # forward propagation on training set
            h = self.model(self.train_graph, self.features)
            train_pos_score = self.pred(self.train_pos_g, h)
            train_neg_score = self.pred(self.train_neg_g, h)

            loss = compute_entropy_loss(train_pos_score, train_neg_score,
                                        self.train_index_dict,
                                        device=self.device, byNode=False)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dur.append(time.time() - cur)
            cur = time.time()

            if e % 100 == 0:
                # evaluation on test set
                self.model.eval()
                with torch.no_grad():
                    test_pos_score = self.pred(self.test_pos_g, h)
                    test_neg_score = self.pred(self.test_neg_g, h)
                    val_loss = compute_entropy_loss(test_pos_score, test_neg_score,
                                        self.test_index_dict,
                                        device=self.device, byNode=False)
                # # print(f"Epoch {e} | Loss{loss.item()} |val loss {val_loss} | Time {dur[-1]}")
                    test_auc, test_ndcg = compute_metric(test_pos_score, test_neg_score,
                                                        self.test_index_dict,
                                                        device=self.device, byNode = True)
                    # train_auc, train_ndcg = compute_metric(train_pos_score, train_neg_score,
                    #                                     train_index_dict,
                    #                                     device=self.device, byNode = eval_by_node)

                print("Epoch {:05d} | Loss {:.4f} | Val loss {:.4f} | Test AUC {:.4f} | Test NDCG {:.4f} | Time {:.4f}".format(
                    e, loss.item(), val_loss.item(), test_auc, test_ndcg, dur[-1]))
        print('========save model=======')
        torch.save(self.model, self.save_path+f'/{self.args.init_dim}_{self.args.dim1}_{self.args.dim2}_epoch{self.args.sage_epoch}.pkl')
        # res =  self.evaluate(self.test_ucands, self.train_graph, self.features, self.test_ur)
        # print(res)
        print('===========before attack=========')
        for i in self.target_items:
            res = self.evaluate(self.train_spy,self.train_graph, self.features, target_item = i)
            print(f'target [{i}] , [{res}]')
        # print(f'hr {hr}, ndcg {ndcg}')
    def train_KGE(self, kg_df, emb_save_path):
        # Load dataset
        kg_object = KnowledgeGraph(kg_df)
        kg_train, kg_val, kg_test = kg_object.split_kg(validation=True)
        # Define some hyper-parameters for training
        emb_dim = 50
        lr = 0.0004
        margin = 0.5
        n_epochs = 1000
        batch_size = 32768
        # Define the model and criterion
        model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel,
                            dissimilarity_type='L2')
        criterion = MarginLoss(margin)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        trainer = Trainer(model, criterion, kg_train, n_epochs, batch_size,
                        optimizer=optimizer, sampling_type='bern', use_cuda='all',)

        trainer.run()

        print('Link prediction Evaluate')
        evaluator = LinkPredictionEvaluator(model, kg_test)
        evaluator.evaluate(200)
        evaluator.print_results()
        
        print('Get entity embedding')
        ent_emb, rel_emb = model.get_embeddings()
        torch.save(ent_emb, emb_save_path)
        _ent_emb = torch.load(emb_save_path)
        print(f'Load ent emb shape = {_ent_emb.shape}')

    def get_target_items(self):
        return self.target_items
    
    def get_meta(self):
        return self.user_num, self.item_num, self.item_num_in_kg, self.target_items, self.kg,\
         self.n_entity, self.n_relation, self.adj_entity, self.adj_relation,self.adj_item_dict,self.popular_items,self.large_degree_items
    
    def attack(self, attack_data, target_item):
        # attack_data to df
        # print('====convert attack profiles to attack df====')
        ui, user, item, r, t = {},[],[],[],[]
        for p in iter(attack_data):
            u = p[0]
            for i in p[1:]:
                user.append(u)
                item.append(i)
                r.append(1.0)
                t.append(1)
        ui['user'] = user
        ui['item'] = item
        ui['rating'] = r
        ui['timestamp'] = t
        attack_df = pd.DataFrame(ui)
        # get train set
        attack_positive = attack_df[['user','item']].reset_index(drop=True)
        # get features
        attack_u = pd.unique(attack_positive['user'])+self.item_num
        # print('======attack users======')
        # print(attack_u)
        features = torch.tensor(self.emb_lookup[list(range(np.max(attack_u+1)))]).to(self.device)
        # G_attack
        src = torch.tensor(attack_positive['user'].to_numpy()).to(self.device)+self.item_num
        dst = torch.tensor(attack_positive['item'].to_numpy()).to(self.device)
        self.train_graph = dgl.add_edges(self.train_graph, src, dst)
        # print('=======attacked graph========')
        # print(self.train_graph)
        # evaluate
        res_train = self.evaluate(self.train_spy, self.train_graph, features, target_item = target_item)
        # print(f'evaluate train spy hr [{hr_train}] ndcg [{ndcg_train}]')
        res_val = self.evaluate(self.eval_spy, self.train_graph, features, target_item = target_item)
        # print(f'evaluate eval spy hr [{hr_eval}] ndcg [{ndcg_eval}]')
        return res_train, res_val

    def evaluate(self, test_ucands, graph, features, test_ur = None, target_item=None):
        assert test_ur is not None or target_item is not None
        # print(f'Start Calculating Metrics......each user have [{self.args.candi_num}] candidates')
        h = self.model(graph, features)
        embeddings = h.detach().cpu().numpy()
        # scores_matrix = np.dot(embeddings, embeddings.T) # row/column = [item_num+user_num+(attack_num)]
        preds = {}
        for u in tqdm(test_ucands.keys()):
            # truncate the acutual candidates num
            if target_item is None:
                test_ucands[u] = test_ucands[u][:self.args.candi_num]
            else:
                test_ucands[u] = test_ucands[u][:self.args.candi_num]+[target_item]
            # pay attention to the u_id here! 
            embeddings[u+self.item_num]
            # y_pred = scores_matrix[u+self.item_num, test_ucands[u]]
            y_pred = np.dot(embeddings[u+self.item_num],embeddings[test_ucands[u]].T)
            # sort y_pred and get the indices
            topk_indices = np.argsort(-y_pred)
            assert y_pred[topk_indices[0]]>y_pred[topk_indices[1]]
            # get the topk item
            preds[u] = np.take(test_ucands[u],topk_indices)
        for u in preds.keys():
            if target_item is None:
                preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]
            else:
                preds[u] = [1 if i==target_item else 0 for i in preds[u]]
        # print('Save metric@k result to res folder...')
        res = pd.DataFrame({'metric@K': ['hr', 'ndcg']})
        hr, ndcg = 0, 0
        for k in [5, 10, 20, 30, 50]:
            if k > self.args.attack_topk:
                continue
            tmp_preds = preds.copy()        
            tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}
            hr_k = hr_at_k(tmp_preds)
            ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])
            # if k == 10:
            #     print(f'HR@{k}: {hr_k:.4f}')
            #     print(f'NDCG@{k}: {ndcg_k:.4f}')
            res[k] = np.array([hr_k, ndcg_k])
            # if k < self.args.attack_topk:
            #     hr, ndcg = hr_k, ndcg_k
        # res.to_csv(
        #     f'{self.result_save_path}/kpi_results.csv', 
        #     index=False
        # )
        return res
    
    def candidate(self, agent_pivot = None):
        # generate valid action set
        candis = []
        # mask = np.array(mask).T

        for b_idx in range(len(self.state)):
            # m = mask[b_idx]
            if self.args.candidate_mode == 'nearest_neighbor':
                pivot = self.state[b_idx][self.step_count]
                pivot_neighbor = self.adj_item_dict[pivot]
                if len(pivot_neighbor) >= self.args.action_size:
                    candis.append(random.sample(pivot_neighbor, self.args.action_size))
                else:
                    rr = self.args.action_size - len(pivot_neighbor)
                    candi = random.sample(list(range(self.item_num_in_kg)), rr) + pivot_neighbor
                    candis.append(candi)
            elif self.args.candidate_mode == 'kg_item':
                candi = random.sample(list(range(self.item_num_in_kg)), self.args.action_size)
                candis.append(candi)
            elif self.args.candidate_mode == 'all_item':
                candi = random.sample(list(range(self.item_num)), self.args.action_size)
                candis.append(candi)
            elif self.args.candidate_mode == 'target_neighbor':
                pivot = self.state[b_idx][1]
                
                pivot_neighbor = self.adj_item_dict[pivot]
                if len(pivot_neighbor) >= self.args.action_size:
                    candis.append(random.sample(pivot_neighbor, self.args.action_size))
                else:
                    rr = self.args.action_size - len(pivot_neighbor)
                    candi = random.sample(list(range(self.item_num_in_kg)), rr) + pivot_neighbor
                    candis.append(candi)
            elif self.args.candidate_mode =='all_hist':
                pivot = self.state[b_idx][1:self.step_count+1]
                # print('pivot',pivot)
                # print('step count', self.step_count)
                all_hist = []
                for p in pivot:
                    all_hist.extend(self.adj_item_dict[p])
                    # print(f'len candi all hist = {len(candi)}')
                if len(all_hist) >= self.args.action_size:
                    candis.append(random.sample(all_hist, self.args.action_size))
                else:
                    rr = self.args.action_size - len(all_hist)
                    candi = random.sample(list(range(self.item_num_in_kg)), rr) + all_hist
                    candis.append(candi)
            elif self.args.candidate_mode == 'agent_select':
                assert agent_pivot is not None
                pivot = self.state[b_idx][agent_pivot[b_idx]]
                pivot_neighbor = self.adj_item_dict[pivot]
                if len(pivot_neighbor) >= self.args.action_size:
                    candis.append(random.sample(pivot_neighbor, self.args.action_size))
                else:
                    rr = self.args.action_size - len(pivot_neighbor)
                    candi = random.sample(list(range(self.item_num_in_kg)), rr) + pivot_neighbor
                    candis.append(candi)
            # candi = set(tmp) - set(m)
        # print('candis',np.array(candis))
        return np.array(candis)
    def p_mask(self):
        if random.random() < self.args.bandit_ratio:
            mask = [[True]*(self.step_count+1) + [False]*(self.episode_length-self.step_count-1)]*len(self.attackers_id)
        else:
            mask = [[True] + [False]*(self.episode_length-1)]*len(self.attackers_id)
        return mask
    '''Gym environment API'''
    def reset(self, target_item, episode):
        # if random.random() < self.args.target_bandit:
        #     self.fisrt_item = target_item
        # else:
        #     self.first_item = random.choice(list(range(self.item_num_in_kg)))
        self.target_item = target_item
        self.attackers_id = [i for i in range(self.user_num+episode*self.args.num_train_attacker, self.user_num+(episode+1)*self.args.num_train_attacker)]
        self.init_item = [target_item for _ in range(len(self.attackers_id))]
        # self.history_items = set()
        # np.array(self.attackers_id)[:, None] # (num_train_attacker,) -> (num_train_attacker,1)
        self.state = np.hstack((np.array(self.attackers_id)[:, None], np.array(self.init_item)[:,None])) # (num_train_attacker,2)
        self.state = np.pad(self.state, ((0,0),(0,self.episode_length - 1)), constant_values = -1)
        self.step_count = 1
        # self.pivot_idx = np.array(self.attackers_id)[:, None] 
        # self.state = np.tile(self.state, 1)[:,None]
        # state = copy.deepcopy(self.state)
        # State dict
        print(f'action candidates mode [{self.args.candidate_mode}], action size = [{self.args.action_size}]')
        
        if self.args.candidate_mode == 'agent_select':
            agent_pivot = [self.step_count for _ in range(len(self.attackers_id))] # target item as the initial pivot
            self.valid_actions = self.candidate(agent_pivot) # the neighbors of pivot as the valid actions

            self.pivot_mask = self.p_mask() # batch_size * mask. the target_item and the second item could be the pivot
            return {'some_state':torch.LongTensor(copy.deepcopy(self.state)),
                 'valid_actions': torch.LongTensor(copy.deepcopy(self.valid_actions)),
                    'pivot_mask': torch.LongTensor(copy.deepcopy(self.pivot_mask))
                    }
        else:
            self.valid_actions = self.candidate()
            return {'some_state':torch.LongTensor(copy.deepcopy(self.state)),
                 'valid_actions': torch.LongTensor(copy.deepcopy(self.valid_actions))
                    }
        
    
    '''Gym environment API'''
    def step(self, item_action, pivot_action):
        '''
        input: 
            action_chosen: [B] 每一行都有一个新的pivot(index)
            action_chosen: [B] 每一行都有一个新的action(item id)
        return 
            reward: [bs*attack_user_num] 到了episode长度时，attack_user_num个profile共同得到一个reward，一共有bs个reward，并需要normalize处理，然后repeat
            done: 到了episode长度时，返回True，否则返回False
        '''
        # a1 = np.expand_dims(a1,1)
        # action index
        self.step_count += 1
        if pivot_action is None:
            action = self.valid_actions[range(self.valid_actions.shape[0]), item_action.cpu().numpy()]
            self.state[:,self.step_count] = action
            self.valid_actions = self.candidate()
        else:
            # print('pivot_action',pivot_action)
            action = self.valid_actions[range(self.valid_actions.shape[0]), item_action.cpu().numpy()]
            self.state[:,self.step_count] = action # update state must before self.candidate(pivot_action) 
            self.valid_actions = self.candidate(pivot_action.cpu().numpy()+1)
            # the pivot mask only affect when mode == 'agent_pivot'
            self.pivot_mask = self.p_mask()
        # action = np.expand_dims(action, 1)
        
        # 生成new_state
        # self.pivot_idx = np.hstack((self.pivot_idx, a1))
        # new_pivot_idx = copy.deepcopy(self.pivot_idx)
        # self.state = np.hstack((self.state, action))
        
        if self.step_count == self.episode_length:
            # rewards = []
            # attack_set = []
            # for i in range(len(self.attackers_id)):
            #     # u_id = self.state[i][0]
            #     user_profile = self.state[i]
            #     attack_set.append(user_profile)
                
            # attack_dataset 传入到RecSys中
            # # print('attack_set = ',attack_set)
            res_train, res_val = self.attack(self.state, self.target_item)
            
            if self.args.rl_reward_type == 'hr':
                # rewards.append(hr)
                rew = res_train.loc[0,self.args.reward_topk]
            elif self.args.rl_reward_type == 'ndcg':
                # rewards.append(ndcg)
                rew = res_train.loc[1,self.args.reward_topk]
            

            user_rewards = []
            # for t in rewards:
            user_rewards.extend([rew for j in range(len(self.attackers_id))])
            reward = np.array(user_rewards)
            
            # reward = user_rewards
            # # print('======reward shape======')
            # # print(reward.shape)
            # # print('=====intrisic_reward=====')
            # # print(new_pivot_idx[:,1:])
            # # print(new_pivot_idx.shape)
            # intrisic_reward = 1/2**np.log(np.max(new_pivot_idx,axis=1)+1)
            # intrisic_reward = np.zeros(new_pivot_idx.shape[0])
            # for i in range(new_pivot_idx.shape[0]):
            #     intrisic_reward[i] = np.count_nonzero(new_pivot_idx[i]==0)
            # from scipy.special import softmax
            # intrisic_reward = softmax(intrisic_reward)
            # # print(intrisic_reward)
            # # print(intrisic_reward)
            
            done = [True]*len(self.attackers_id)

        else:
            res_train = -1
            res_val = -1
            # intrisic_reward = np.zeros(len(self.attackers_id))
            reward = np.zeros(len(self.attackers_id))
            done = [False]*len(self.attackers_id)
        
        
        info = {'res_train':res_train, 'res_val':res_val }
        if pivot_action is None:
            return {'some_state':torch.LongTensor(copy.deepcopy(self.state)),
                    'valid_actions': torch.LongTensor(copy.deepcopy(self.valid_actions))
                    }, reward, done, info
        else:
            return {'some_state':torch.LongTensor(copy.deepcopy(self.state)),
                    'valid_actions': torch.LongTensor(copy.deepcopy(self.valid_actions)),
                    'pivot_mask': torch.LongTensor(copy.deepcopy(self.pivot_mask))
                    }, reward, done, info
        # return new_state, reward, done info
            
if __name__=="__main__":
    from collections import defaultdict
    target_items_results_hr_5 = defaultdict(list)
    target_items_results_hr_10 = defaultdict(list)
    target_items_results_hr_20 = defaultdict(list)
    target_items_results_ndcg_5 = defaultdict(list)
    target_items_results_ndcg_10 = defaultdict(list)
    target_items_results_ndcg_20 = defaultdict(list)

    args = parse_args()
    set_global_seeds(args.seed)
    env = Env(args)
    target_items = env.get_target_items()
    print('===========target_items==========')
    print(target_items)
    n_users, n_items, _, _, _, _,_,_,_, adj_item_dict,popular_items,large_degree_items= env.get_meta()
    episode_length = args.episode_length # 16:hr0.32, 32:hr0.42, 64:hr0.9
    eval_n_attacker = 3
    sample_times = 50
    alpha = args.action_size    # 1000 ok 500变弱， 1500变弱
    if args.random_method == 'n':
        method = 'target_neighbor_attack'
    elif args.random_method == 'r':
        method = 'target_random_attack'
    elif args.random_method == 'rn':
        method = 'neighbor_random_attack'
    elif args.random_method == 'd':
        method = 'degree_attack'
    for tar in range(len(target_items)):
        set_global_seeds(args.seed) # seed should be here, before each Env init
        env = Env(args)
        # if tar in [0,1,2,3,4,5,6]: continue
        target_item = target_items[tar]
        # print(f'target_item is {target_item}')
        # print('*'*50)
        num_users = n_users
        pivot = target_item
        # env.reset_model()
        final_results_train = {'hr':{5:[],10:[],20:[]}, 'ndcg':{5:[],10:[],20:[]}}
        final_results_val = {'hr':{5:[],10:[],20:[]}, 'ndcg':{5:[],10:[],20:[]}}
        # item_set_random = random.sample(list(range(n_items)),alpha)
        for s in range(sample_times):
            attack_data = []
            for i in tqdm(range(num_users, num_users+eval_n_attacker),desc='num attacker'):
                attack_profile = [i]
                if random.random() < args.target_bandit:
                    attack_profile.append(target_item)
                else:
                    attack_profile.append(random.choice(list(range(n_items))))
                while(len(attack_profile)-1<=episode_length):
                    # item_set = random.sample(list(range(n_items)),alpha)
                    if args.random_method == 'n':
                        pivot_list = adj_item_dict[pivot]
                        # item_set = random.sample(pivot_list,alpha)
                        item_set = pivot_list
                        j2 = random.choice(item_set)
                        # pivot = j2
                    elif args.random_method == 'r':
                        item_set_random = list(range(n_items))
                        j2 = random.choice(item_set_random)
                    elif args.random_method == 'p':
                        item_set = popular_items
                        j2 = random.choice(item_set)
                    elif args.random_method == 'd':
                        if alpha<=len(large_degree_items):
                            print('no plus')
                            item_set = random.sample(large_degree_items,alpha)
                        else:
                            print('no plus')
                            item_set = random.sample(large_degree_items,alpha - len(large_degree_items)) + large_degree_items
                        # item_set = random.sample(list(range(n_items)),alpha) + large_degree_items
                        j2 = random.choice(item_set)
                    else:
                        if alpha<=len(adj_item_dict[pivot]):
                            print('no plus')
                            item_set = random.sample(adj_item_dict[pivot],alpha)
                        else:
                            print('no plus')
                            item_set = random.sample(list(range(n_items)),alpha - len(adj_item_dict[pivot])) + adj_item_dict[pivot]
                        # item_set = random.sample(list(range(n_items)),alpha) + adj_item_dict[pivot]
                        j2 = random.choice(item_set)
                    attack_profile.append(j2)
                    # item_set.remove(j2)
                attack_data.append(attack_profile)
            res_train, res_val = env.attack(attack_data, target_item)
            for k in [5, 10, 20]:
                final_results_train['hr'][k].append(res_train.loc[0,k])
                final_results_train['ndcg'][k].append(res_train.loc[1,k])
                final_results_val['hr'][k].append(res_val.loc[0,k])
                final_results_val['ndcg'][k].append(res_val.loc[1,k])
            num_users = num_users+eval_n_attacker # update attacker's id
        # print(result_dict)
        target_items_results_hr_5[target_item] = final_results_val['hr'][5]
        target_items_results_ndcg_5[target_item] = final_results_val['ndcg'][5]
        target_items_results_hr_10[target_item] = final_results_val['hr'][10]
        target_items_results_ndcg_10[target_item] = final_results_val['ndcg'][10]
        target_items_results_hr_20[target_item] = final_results_val['hr'][20]
        target_items_results_ndcg_20[target_item] = final_results_val['ndcg'][20]
        del env
    # HR
    target_items_results_hr5_df = pd.DataFrame(target_items_results_hr_5)
    target_items_results_hr5_df[-1] = target_items_results_hr5_df.mean(1)
    target_items_results_hr10_df = pd.DataFrame(target_items_results_hr_10)
    target_items_results_hr10_df[-1] = target_items_results_hr10_df.mean(1)
    target_items_results_hr20_df = pd.DataFrame(target_items_results_hr_20)
    target_items_results_hr20_df[-1] = target_items_results_hr20_df.mean(1)
    # NDCG
    target_items_results_ndcg_5 = pd.DataFrame(target_items_results_ndcg_5)
    target_items_results_ndcg_5[-1] = target_items_results_ndcg_5.mean(1)
    target_items_results_ndcg_10 = pd.DataFrame(target_items_results_ndcg_10)
    target_items_results_ndcg_10[-1] = target_items_results_ndcg_10.mean(1)
    target_items_results_ndcg_20 = pd.DataFrame(target_items_results_ndcg_20)
    target_items_results_ndcg_20[-1] = target_items_results_ndcg_20.mean(1)
    print(f'=====avg of each iteration method {args.random_method} action_size {args.action_size} seed {args.seed}=======')
    print('top20')
    print(target_items_results_hr20_df[-1])
    print('top10')
    print(target_items_results_hr10_df[-1])
    print('top5')
    print(target_items_results_hr5_df[-1])
    print('ndcg20')
    print(target_items_results_ndcg_20[-1])
    print('ndcg10')
    print(target_items_results_ndcg_10[-1])
    print('ndcg5')
    print(target_items_results_ndcg_5[-1])
    
    if args.random_method == 'r':
        target_items_results_hr20_df.to_csv(f'result_{args.dataset}/{method}/seed_{args.seed}_epi{episode_length}_att{eval_n_attacker}_dim{args.init_dim}{args.dim1}{args.dim2}_kghop{args.kg_hop}.csv', index=False)
    else:
        target_items_results_hr20_df.to_csv(f'result_{args.dataset}/{method}/seed_{args.seed}_action_size_{args.action_size}_epi{episode_length}_att{eval_n_attacker}_dim{args.init_dim}{args.dim1}{args.dim2}_kghop{args.kg_hop}.csv', index=False)
    # kg neighbor attack