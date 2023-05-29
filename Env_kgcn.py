import numpy as np
import copy
import json
import pickle
import time
import pandas as pd
import os
import random
from tqdm import tqdm
import torch
from Args import parse_args
from daisy.utils.loader import load_rate, get_ur, build_candidates_set, load_attack_data_get_attack_instances
from daisy.utils.map import neaten_id, get_unpopular, find_neighbor_items
from daisy.utils.splitter import split_test
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torchkge.data_structures import KnowledgeGraph
from torchkge.evaluation import LinkPredictionEvaluator,TripletClassificationEvaluator
from torchkge.models import TransEModel
from torchkge.utils import Trainer, MarginLoss
import torch.optim as optim
from kgcn.model import KGCN
from sklearn.metrics import roc_auc_score

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)

# Dataset class
class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['user'], dtype= np.long)
        item_id = np.array(self.df.iloc[idx]['item'], dtype = np.long)
        label = np.array(self.df.iloc[idx]['rating'], dtype=np.float32)
        return user_id, item_id, label

class Env():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu" )
        self._data_path = f'data/ds/{args.dataset}'
        self.save_path = f'{args.processed_path}_{args.dataset}'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.episode_length = args.episode_length
        df, self.user_num, self.item_num = load_rate(args.dataset, args.prepro, binary = True, level = args.level)
        df, self.item_num_in_kg, self.n_entity, self.n_relation, self.kg, self.kg_dict, self.adj_entity, self.adj_relation, self.degree_item = neaten_id(args.dataset, df, args.kg_neighbor_size)
        self.degree = list(dict(sorted(self.degree_item.items(), key=lambda item: item[1],reverse=True)).keys())
        self.large_degree_items = self.degree[:int(len(self.degree)*0.3)]
        # Unpopular + popular
        unpopular_items,self.popular_items = get_unpopular(df, self.item_num_in_kg, args.unpop_ratio)        
        # item pool
        self.item_pool = set(range(self.item_num))
        # target_item
        self.target_items = random.sample(unpopular_items, self.args.num_target_item)
        print(f'target items = {self.target_items}')
        # adj_item_dict
        self.adj_item_dict = find_neighbor_items(f'{self._data_path}/adj_item_{self.args.prepro}_{self.args.level}_{self.args.kg_hop}.pkl', self.adj_entity, self.item_num_in_kg, self.args.kg_hop)
        # generate KGE
        KGFILE = {'ml-1m':'ml1m-kg1m','ml-20m':'ml20m-kg500k','bx':'bx-kg150k','lastfm':'lastfm-kg15k'}
        if not os.path.exists(f'data/kg/{KGFILE[args.dataset]}/ent_embedding.pt'):
            print('============KGE============')
            kg_df = pd.DataFrame(self.kg, columns=['from', 'rel', 'to'])
            kg_df = kg_df.reindex(columns = ['from','to','rel'])
            self.train_KGE(kg_df,f'data/kg/{KGFILE[args.dataset]}/ent_embedding.pt')
        # ==============negative sampling=================(From KGCN)
        # print('negative sampling for x_train_positve')
        # user_list = []
        # item_list = []
        # label_list = []
        # for user, group in df.groupby(['user']): # sample the same size of negative item with positive item.
        #     item_set = set(group['item'])
        #     negative_set = self.item_pool - item_set
        #     negative_sampled = random.sample(negative_set, len(item_set))
        #     user_list.extend([user]*len(negative_sampled))
        #     item_list.extend(negative_sampled)
        #     label_list.extend([0]*len(negative_sampled))
        # negative = pd.DataFrame({'user':user_list, 'item':item_list, 'rating':label_list}) 
        # df_dataset = pd.concat([df, negative])
        # df_dataset = df_dataset.sample(frac=1,replace = False, random_state = args.seed)
        # df_dataset.reset_index(inplace = True, drop=True)
        # x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['rating'], test_size=args.test_size, shuffle=False, random_state=args.seed)
        # SPLIT DATASET BASED ON USER
        x_train_positive, x_test_positive = split_test(df, 'ufo', self.args.test_size)
        self.test_ur = get_ur(x_test_positive)
        self.total_train_ur = get_ur(x_train_positive)
        self.test_ucands, self.train_spy, self.eval_spy = build_candidates_set(self.test_ur, self.total_train_ur, self.item_pool, self.args, self.target_items)
        # ==============split the data to train and test dataset==================
        # x_train_positive, x_test_positive, y_train, y_test = train_test_split(df, df['rating'], test_size=args.test_size, shuffle=False, random_state=args.seed)
        # get test user profile and random select subset of them as train spy and eval spy.
        # self.test_ur = get_ur(x_test_positive)
        # self.test_ucands, self.train_spy, self.eval_spy = build_candidates_set(self.test_ur, get_ur(x_train_positive), self.item_pool, args, self.target_items)
        # ==============negative sampling=================(From KGCN)
        print('negative sampling for x_train_positve')
        user_list = []
        item_list = []
        label_list = []
        for user, group in x_train_positive.groupby(['user']): # sample the same size of negative item with positive item.
            item_set = set(group['item'])
            negative_set = self.item_pool - item_set
            negative_sampled = random.sample(negative_set, 2*len(item_set))
            user_list.extend([user]*len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0]*len(negative_sampled))
        negative = pd.DataFrame({'user':user_list, 'item':item_list, 'rating':label_list}) 
        x_train = pd.concat([x_train_positive, negative])
        x_train = x_train.sample(frac=1,replace = False, random_state = args.seed)
        x_train.reset_index(inplace = True, drop=True)

        print('negative sampling for x_test_positve')
        user_list = []
        item_list = []
        label_list = []
        for user, group in x_test_positive.groupby(['user']): # sample the same size of negative item with positive item.
            item_set = set(group['item'])
            negative_set = self.item_pool - item_set
            negative_sampled = random.sample(negative_set, len(item_set))
            user_list.extend([user]*len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0]*len(negative_sampled))
        negative = pd.DataFrame({'user':user_list, 'item':item_list, 'rating':label_list}) 
        x_test = pd.concat([x_test_positive, negative])
        x_test = x_test.sample(frac=1,replace = False, random_state = args.seed)
        x_test.reset_index(inplace = True, drop=True)

        train_dataset = KGCNDataset(x_train)
        test_dataset = KGCNDataset(x_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
        # prepare network, loss function, optimizer
        # Knowledge graph is dictionary form 'head': [(relation, tail), ...]
        self.net = KGCN(self.user_num+self.args.num_max_attacker, self.n_entity, self.n_relation, self.item_num, self.item_num_in_kg, self.kg_dict, self.args, self.adj_entity, self.adj_relation, self.device).to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.kgcn_lr, weight_decay=self.args.kgcn_l2_weight)
        # train (use train positive and train negative)
        loss_list = []
        test_loss_list = []
        auc_score_list = []
        self.result_save_path = f'{self.save_path}/kgcn/'
        # load model
        if os.path.exists(self.result_save_path+f'/model_{self.args.kgcn_dim}_{self.args.kgcn_aggregator}_{self.args.kgcn_epochs}_{self.args.kgcn_n_iter}_{self.args.kgcn_l2_weight}.pth'):
            self.net.load_state_dict(torch.load(self.result_save_path+f'/model_{self.args.kgcn_dim}_{self.args.kgcn_aggregator}_{self.args.kgcn_epochs}_{self.args.kgcn_n_iter}_{self.args.kgcn_l2_weight}.pth', map_location = 'cuda:0'))
        # train model
        else:
            for epoch in range(self.args.kgcn_epochs):
                running_loss = 0.0
                for i, (user_ids, item_ids, labels) in enumerate(tqdm(train_loader)):
                    user_ids, item_ids, labels = user_ids.to(self.device), item_ids.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.net(user_ids, item_ids)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                
                # print train loss per every epoch
                print('[Epoch {}]train_loss: '.format(epoch+1), running_loss / len(train_loader))
                loss_list.append(running_loss / len(train_loader))
                    
                # evaluate per every epoch (use test positive and test negative)
                with torch.no_grad():
                    test_loss = 0
                    total_roc = 0
                    for i, (user_ids, item_ids, labels) in enumerate(tqdm(test_loader)):
                        user_ids, item_ids, labels = user_ids.to(self.device), item_ids.to(self.device), labels.to(self.device)
                        outputs = self.net(user_ids, item_ids)
                        test_loss += self.criterion(outputs, labels).item()
                        total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    print('[Epoch {}]test_loss: '.format(epoch+1), test_loss / len(test_loader))
                    print('[Epoch {}]test_roc: '.format(epoch+1), total_roc / len(test_loader))
                    test_loss_list.append(test_loss / len(test_loader))
                    auc_score_list.append(total_roc / len(test_loader))
            # save the model
            if not os.path.exists(self.result_save_path):
                os.makedirs(self.result_save_path)
            torch.save(self.net.state_dict(), self.result_save_path+f'/model_{self.args.kgcn_dim}_{self.args.kgcn_aggregator}_{self.args.kgcn_epochs}_{self.args.kgcn_n_iter}_{self.args.kgcn_l2_weight}.pth')
    def get_target_items(self):
        return self.target_items
    
    def get_meta(self):
        return self.user_num, self.item_num, self.item_num_in_kg, self.target_items, self.kg,\
         self.n_entity, self.n_relation, self.adj_entity, self.adj_relation,self.adj_item_dict,self.popular_items,self.large_degree_items
        # res = self.evaluate(self.test_ucands,test_ur = self.test_ur)
        # print(res)
        #print(f'evaluate train spy hr [{hr_train}] ndcg [{ndcg_train}]')
        

        # dataset records the rating information. user item and label.
        # 3 things are critical: ratings, user id, and item id:
        # 1. ratings contains the positive and negative samples
        # 2. user id contains the id start from 0
        # 3. item id contains the id start from 0. The item id is entity id in the source code of KGCN, but here we extend it to all the possible items.

        # we first assign the user id and item id, then deal with the negative samples.
        # 1. get the pre-processed dataset.
        # Line 20
        # 2. The ID of the pre-processed dataset should be neaten. The user id is easy to process since we can use the followings:
        # original_user = rating_df['user'].unique()
        # user map = { user: idx for idx, user in enumerate(original items) }
        # rating_df['user'] = rating_df['user'].apply(lambda user:user_map[user])
        # The item id is hard to process since it connects with the knowledge graph.
        # We have the 'item_id2entity_id' information. Thus we can first give each line an ID start from zero. Then the entity and item in 'item_id2entity_id'
        # will have the consistent id. This can be achieved in line 18-31 in map.py. We first get the entire item-ID from the pre-processed dataset and 
        # filter out the item-IDs in 'item_id2entity_id' that are not appeared in the pre-processed dataset. Then read the lines in 'item_id2entity_id'
        # sequentially to get the new id. (we should care about the format of the ID, e.g., str or int).
        # Finally, we will get two maps 'item_index_old2new' and 'entity_id2index'
        # 3. Then, having this consistent map information, we will map the item-ID in dataset and entity-ID in kg to new ID.
        # if item id not in the 'item_index_old2new', then gives such item a new id. The process is the same for entities.
        # 4. having the entity id and relational id, we should construct a undirected KG stored by following lines:
        #  for head_id, relation_id, tail_id in kg:
        #    kg_dict[head_id].append((relation_id, tail_id))
        # Then adj_entity and adj_relation is constructed by the following:
        # Give neighbor size, sample samples from kg_dict[entity], and then construct two Adjencent List respectively.
        
        # To adapt KGCN, we first see how KGCN process the dataset.
        # it has three dataset: rating, KG triples, and a map files.
        # it first filter out the ratings records whose item is not appeared in the map file. 
        # Then it encodes the user ID, entity_ID(entity in map files, head, tail) and relations
        # Then for item ID in the rating file, the item ID first mapped to the entity ID, then give the mapped entity ID a new entity ID.
        # The label are transformed to implicit format.
        # Perform negative sampling for each user: 
        #   for user, group in df_dataset.groupby(['userID']):
        #       item_set = set(group['itemID'])
        #       negative_set = [total item set ] - interated item set
        # [total item set] is very important! In original KGCN, it utilizes the entire entity set, then after negative sampling, the negative items 
        # will have many entities. 
        # Then in the forward() of KGCN, given user-item pair, the item embedding will aggregate its neighbor informaiton. and then calculate the scores
        # between user-item. (user embedding are random initialized)
        
        # A. we maintain the preprocessed dataset and neaten the ID of KG and items like before.
        # B. Get negative samples and set [total item set] as the entire items. 
        # C. Get the adj_entity, adj_relation each entity. For each item, we find its neighbor entity (this will detailed in the next paragraph.)
        # if the item is not in KG, then its neighbor entity will be itself.  
        # D. In KGCN, given the u-v pair from the rating, we get neighbors of v according to its adj_entity, if it is not an entity
        # we assign his neighbors to it. (this will detailed in the next paragraph.)
        # E. What is the details of the aggregate? since it is critical to use embedding instead of the ID to perform aggregate:
        # Given target item v, the self._get_neighbors(v) will generate hop neighbors, which is stored in a List[List]. List[0],...,List[hop] dentoes
        # hop neighbors respectively. The same thing happed for relations.
        # Then, given entities, relations, and u, we utilize self._aggregate(u, entities, relations)
        # We first intialize 3 Look-up Embedding Table for user, entity, relations
        # Then, we first get the embedding for each hop neighbor entities and relations, and get the embedding for the user.
        # Next, our procedure will be: List[0]<--List[1], List[1]<---List[2],...List[hop-1]<---List[hop]
        # And, List[0]<--List[1], List[1]<---List[2],...List[hop-2]<---List[hop-1]
        # Finally, we will get the List[0] representation, which will be the item embedding.
        
        # To extend KGCN, given the u-v pair, if v is in KG, then it will have entitiy neighbors. But if v is not in KG, it will not have entity neighbors,
        # We have to give such v dummy neighbors, whose embedding is a zero vector. 
        # To give dummy neighbors, in KGCN.py, we should edit the self._get_neighbors(). Specifically, 
        # entities = [v] is a batch of seed items(entity or non-entity). self.adj_ent[[v]] will get the neighbors of the entity.
        # Here is very important, since there are mutual ID in entity set and item set, if we use this to get the neighbors of the entity, then those items
        # who do not have neigbors will get neighbors. Therefore, we need to add a judgment here. 
        # for vv in v: if v in list(range(items in kg)): neighbor_entities.append(self.adj_ent[v])
        # else:  neighbors_entities.extend(pre-defined dummy node, (e.g., num_KG_entities) )
        # In this way, for all batch nodes v, we will construct the neighbor entities. The same thing happend for the relation.
    
    def attack(self, attack_data, target_item, eval=False):
        # attack_data to df
        print('====convert attack profiles to attack df====')
        ui, user, item, r, t = {},[],[],[],[]
        for p in attack_data:
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
        # negative sampling
        print('negative sampling')
        full_item_set = set(range(self.item_num))
        user_list = []
        item_list = []
        label_list = []
        for user, group in attack_df.groupby(['user']): # for each user, sample a negative item for each positive item
            item_set = set(group['item'])
            negative_set = full_item_set - item_set
            negative_sampled = random.sample(negative_set, len(item_set))
            user_list.extend([user]*len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0]*len(negative_sampled))
        negative = pd.DataFrame({'user':user_list, 'item':item_list, 'rating':label_list}) # label?
        attack_df = pd.concat([attack_df, negative])
        attack_df = attack_df.sample(frac=1,replace = False, random_state = self.args.seed)
        attack_df.reset_index(inplace = True, drop=True)
        # dataloader
        train_dataset = KGCNDataset(attack_df)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size)
        for epoch in range(self.args.attack_epochs):
            running_loss = 0.0
            for i, (user_ids, item_ids, labels) in enumerate(tqdm(train_loader)):
                user_ids, item_ids, labels = user_ids.to(self.device), item_ids.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(user_ids, item_ids)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                self.optimizer.step()

                running_loss += loss.item()
        # evaluate on spy user
        res_train = self.evaluate(self.train_spy, target_item = target_item)
        #print(f'evaluate train spy hr [{hr_train}] ndcg [{ndcg_train}]')
        res_val = self.evaluate(self.eval_spy, target_item = target_item)
        #print(f'evaluate eval spy hr [{hr_eval}] ndcg [{ndcg_eval}]')
        return res_train, res_val
    
    def evaluate(self, test_ucands, test_ur = None, target_item=None):
        assert test_ur is not None or target_item is not None
        # print(f'Start Calculating Metrics......each user have [{self.args.candi_num}] candidates')
        preds = {}
        for u in tqdm(test_ucands.keys()):
            # truncate the acutual candidates num
            if target_item is None:
                test_ucands[u] = test_ucands[u][:self.args.candi_num]
            else:
                test_ucands[u] = test_ucands[u][:self.args.candi_num]+[target_item]
            # build a test MF dataset for certain user u to accelerate
            tmp = pd.DataFrame({
                'user': [u for _ in test_ucands[u]], 
                'item': test_ucands[u], 
                'rating': [0. for _ in test_ucands[u]], # fake label, make nonsense
            })
            tmp_dataset = KGCNDataset(tmp)
            tmp_loader = torch.utils.data.DataLoader(tmp_dataset, batch_size=len(tmp))
            # find the top k item for user u based on similarity
            with torch.no_grad():
                test_loss = 0
                total_roc = 0
                for i, (user_ids, item_ids, labels) in enumerate(tmp_loader):
                    user_ids, item_ids, labels = user_ids.to(self.device), item_ids.to(self.device), labels.to(self.device)
                    similarities = self.net(user_ids, item_ids)
            _, topk_indices = torch.topk(similarities, self.args.topk)
            # topk_indices = similarities.sort() given self.args.topk
            preds[u] = torch.take(torch.tensor(test_ucands[u]), topk_indices.cpu()).numpy()
        # =======================calculate HR and NDCG===============================
        # preds is a dict contains the topk item for each user u
        for u in preds.keys():
            if target_item is None:
                preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]
            else:
                preds[u] = [1 if i==target_item else 0 for i in preds[u]]
            # process topN list and store result for reporting KPI
        print('Save metric@k result to res folder...')
        res = pd.DataFrame({'metric@K': ['hr', 'ndcg']})
        hr, ndcg = 0, 0
        for k in [1, 5, 10, 20, 30, 50]:
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
            # if k == self.args.attack_topk:
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
        return np.array(candis)
    def p_mask(self):
        if random.random() < self.args.bandit_ratio:
            mask = [[True]*(self.step_count+1) + [False]*(self.episode_length-self.step_count-1)]*len(self.attackers_id)
        else:
            mask = [[True] + [False]*(self.episode_length-1)]*len(self.attackers_id)
        return mask
    '''Gym environment API'''
    def reset(self, target_item, episode):
        self.target_item = target_item
        self.attackers_id = [i for i in range(self.user_num+episode*self.args.num_train_attacker, self.user_num+(episode+1)*self.args.num_train_attacker)]
        self.init_item = [target_item for _ in range(len(self.attackers_id))]
        self.state = np.hstack((np.array(self.attackers_id)[:, None], np.array(self.init_item)[:,None])) # (num_train_attacker,2)
        # Pad state to keep dimension consistent
        self.state = np.pad(self.state, ((0,0),(0,self.episode_length - 1)), constant_values = -1)
        self.step_count = 1
        # State dict
        print(f'action candidates mode [{self.args.candidate_mode}], action size = [{self.args.action_size}]')
        if self.args.candidate_mode == 'agent_select':
            agent_pivot = [1 for _ in range(len(self.attackers_id))] # target item as the initial pivot
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
            action_chosen: [B] 每一行都有一个新的action(item id)
        return 
            reward: [bs*attack_user_num] 到了episode长度时，attack_user_num个profile共同得到一个reward，一共有bs个reward，并需要normalize处理，然后repeat
            done: 到了episode长度时，返回True，否则返回False
        '''
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

        if self.step_count == self.episode_length:
            res_train, res_val = self.attack(self.state, self.target_item)
            if self.args.rl_reward_type == 'hr':
                rew = res_train.loc[0,self.args.reward_topk]
            elif self.args.rl_reward_type == 'ndcg':
                rew = res_train.loc[1,self.args.reward_topk]
            user_rewards = []
            user_rewards.extend([rew for j in range(len(self.attackers_id))])
            reward = np.array(user_rewards)
            done = [True]*len(self.attackers_id)
        else:
            res_train = -1
            res_val = -1
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

if __name__ == '__main__':
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
    n_users, n_items, _, _, _, _,_,_,_, adj_item_dict, popular_items, large_degree_items= env.get_meta()
    episode_length = args.episode_length # 16:hr0.32, 32:hr0.42, 64:hr0.9
    eval_n_attacker = args.num_train_attacker
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
        set_global_seeds(args.seed)
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
                attack_profile.append(target_item)
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
                    # elif args.random_method == 'p':
                    #     item_set = popular_items
                    #     j2 = random.choice(item_set)
                    # elif args.random_method == 'd':
                    #     if alpha<=len(large_degree_items):
                    #         print('no plus')
                    #         item_set = random.sample(large_degree_items,alpha)
                    #     else:
                    #         print('no plus')
                    #         item_set = random.sample(large_degree_items,alpha - len(large_degree_items)) + large_degree_items
                    #     # item_set = random.sample(list(range(n_items)),alpha) + large_degree_items
                    #     j2 = random.choice(item_set)
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
            num_users = num_users+eval_n_attacker
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
    print(f'============avg of each iteration method {args.random_method} action_size {args.action_size} seed {args.seed}============')
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


