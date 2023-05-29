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
import copy
import json
import pickle
import time
from tqdm import tqdm
import pandas as pd

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)

class Env():
    def __init__(self, args):
        # set_global_seeds(28)
        self.args = args
        self.device = torch.device('cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() else "cpu" )
        self._data_path = f'data/ds/{self.args.dataset}'
        self.save_path = f'{self.args.processed_path}_{self.args.dataset}'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # attack parameters
        self.episode_length = self.args.episode_length
        # preprocess dataset
        df, self.user_num, self.item_num = load_rate(self.args.dataset, self.args.prepro, binary=True, level=self.args.level)
        # clean id + KG
        df, self.item_num_in_kg, self.n_entity, self.n_relation, self.kg, self.kg_dict, self.adj_entity, self.adj_relation, self.degree_item = neaten_id(self.args.dataset, df, self.args.kg_neighbor_size)
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
        unpopular_items,self.popular_items = get_unpopular(df, self.item_num_in_kg, args.unpop_ratio)        
        # item pool
        self.item_pool = set(range(self.item_num))
        # target_item
        self.target_items = random.sample(unpopular_items, self.args.num_target_item)
        print(f'target items = {self.target_items}')
        # adj_item_dict
        self.adj_item_dict = find_neighbor_items(f'{self._data_path}/adj_item_{self.args.prepro}_{self.args.level}_{self.args.kg_hop}.pkl', self.adj_entity, self.item_num_in_kg, self.args.kg_hop)
        # Negative sampler
        self.spler = Sampler(
            self.user_num+self.args.num_max_attacker, 
            self.item_num, 
            num_ng=self.args.num_ng, 
            sample_method=self.args.sample_method, 
            sample_ratio=self.args.sample_ratio
        )
        # Train, Val, Test
        # self.processed_path = f'{self.save_path}/'
        # if not os.path.exists(self.processed_path):
        #     os.makedirs(self.processed_path)
        if os.path.exists(f'{self.save_path}/split_{self.args.num_ng}_{self.args.algo_name}_seed{self.args.seed}.pkl'):
            load_time = time.time()
            with open(f'{self.save_path}/split_{self.args.num_ng}_{self.args.algo_name}_seed{self.args.seed}.pkl','rb') as f: 
                process_dict = pickle.load(f)
            self.train_set = process_dict['train_set']
            self.test_set = process_dict['test_set']
            self.test_ur = process_dict['test_ur']
            self.total_train_ur = process_dict['total_train_ur']
            self.test_ucands = process_dict['test_ucands']
            self.train_spy = process_dict['train_spy']
            self.eval_spy = process_dict['eval_spy']
            self.train_neg = process_dict['train_neg']
            self.test_neg = process_dict['test_neg']
            print(f'Load time elapsed= {time.time() - load_time}')
        else:
            self.train_set, self.test_set = split_test(df, 'ufo', self.args.test_size)
            # self.train_set, self.val_set = split_test(self.train_set, self.args.test_method, 0.1)
            self.test_ur = get_ur(self.test_set)
            self.total_train_ur = get_ur(self.train_set)
            self.test_ucands, self.train_spy, self.eval_spy = build_candidates_set(self.test_ur, self.total_train_ur, self.item_pool, self.args, self.target_items)
            self.train_neg = self.spler.transform(self.train_set, is_training=True)
            self.test_neg = self.spler.transform(self.test_set, is_training=True)
            # save!!
            save_dict = {
                'train_set':self.train_set,
                'test_set':self.test_set,
                'test_ur':self.test_ur,
                'total_train_ur':self.total_train_ur,
                'test_ucands':self.test_ucands,
                'train_spy':self.train_spy,
                'eval_spy':self.eval_spy,
                'train_neg':self.train_neg,
                'test_neg':self.test_neg
            }
            with open(f'{self.save_path}/split_{self.args.num_ng}_{self.args.algo_name}_seed{self.args.seed}.pkl', 'wb') as f:
                pickle.dump(save_dict,f)
            
        
        # model
        self.build_model()

    def build_model(self):
        self.result_save_path = f'{self.save_path}/{self.args.algo_name}'+\
                f'{self.args.prepro}_{self.args.level}_{self.args.test_method}_{self.args.epochs}_{self.args.sample_ratio}_{self.args.sample_method}_{self.args.loss_type}_{self.args.problem_type}'+\
                    f'{self.args.dataset}'
        if self.args.problem_type == 'pair':
            train_dataset = PairData(self.train_neg, is_training=True)
        else:
            train_dataset = PointData(self.train_neg, is_training=True)
        if self.args.problem_type == 'pair':
            if self.args.algo_name == 'NeuMF':
                from daisy.model.pair.NeuMFRecommender import PairNeuMF
                print(f'================tareget_model [{self.args.algo_name}_{self.args.problem_type}]===============')
                self.model = PairNeuMF(
                    self.user_num+self.args.num_max_attacker, 
                    self.item_num,
                    factors=self.args.factors,
                    num_layers=self.args.num_layers,
                    q=self.args.dropout,
                    lr=self.args.lr,
                    epochs=self.args.epochs,
                    reg_1=self.args.reg_1,
                    reg_2=self.args.reg_2,
                    loss_type=self.args.loss_type,
                    gpuid=self.device
                )
        elif self.args.problem_type == 'point':
            if self.args.algo_name == 'NeuMF':
                from daisy.model.point.NeuMFRecommender import PointNeuMF
                print(f'================tareget_model [{self.args.algo_name}_{self.args.problem_type}]===============')
                self.model = PointNeuMF(
                    self.user_num+self.args.num_max_attacker, 
                    self.item_num,
                    factors=self.args.factors,
                    num_layers=self.args.num_layers,
                    q=self.args.dropout,
                    lr=self.args.lr,
                    epochs=self.args.epochs,
                    reg_1=self.args.reg_1,
                    reg_2=self.args.reg_2,
                    loss_type=self.args.loss_type,
                    gpuid=self.device
                )
        # 如果有模型，就加载模型
        if os.path.exists(self.result_save_path+f'/model_{self.args.factors}_{self.args.num_layers}_{self.args.dropout}_{self.args.epochs}_{self.args.reg_1}_{self.args.reg_2}.pth'):
            self.model.load_state_dict(torch.load(self.result_save_path+f'/model_{self.args.factors}_{self.args.num_layers}_{self.args.dropout}_{self.args.epochs}_{self.args.reg_1}_{self.args.reg_2}.pth',map_location='cuda:0'))
            self.model.to(self.device)
            # 构造test_ucands
            # model.evaluate_model()
        # 否则，走一遍模型的全流程(earlystop, checkpoint)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=128, 
                shuffle=True, 
                num_workers=4
            )
            if not os.path.exists(self.result_save_path):
                os.makedirs(self.result_save_path)
            s_time = time.time()
            self.model.fit(train_loader)
            elapsed_time = time.time() - s_time
            print(f'prepare the black box model elapsed_time= {elapsed_time}')
            torch.save(self.model.state_dict(), self.result_save_path+f'/model_{self.args.factors}_{self.args.num_layers}_{self.args.dropout}_{self.args.epochs}_{self.args.reg_1}_{self.args.reg_2}.pth')
        
        #res = self.evaluate(self.test_ucands, self.test_ur)
        #print(res)
        # print('===========before attack=========')
        # for i in self.target_items:
        #     res = self.evaluate(self.train_spy, target_item = i)
        #     print(f'target [{i}] , [{res}]')
        
    def reset_model(self):
        if os.path.exists(self.result_save_path):
            self.model.load_state_dict(torch.load(self.result_save_path+'/model.pth'))
            self.model.to(self.device)
    def get_target_items(self):
        return self.target_items
    
    def get_meta(self):
        return self.user_num, self.item_num, self.item_num_in_kg, self.target_items, self.kg,\
         self.n_entity, self.n_relation, self.adj_entity, self.adj_relation,self.adj_item_dict,self.popular_items,self.large_degree_items
    
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
        # get train set
        attack_neg_set = self.spler.transform(attack_df, is_training=True)
        if self.args.problem_type == 'pair':
            attack_dataset = PairData(attack_neg_set, is_training=True)
        else:
            attack_dataset = PointData(attack_neg_set, is_training=True)
        
        attack_loader = torch.utils.data.DataLoader(
                attack_dataset, 
                batch_size=128, 
                shuffle=True, 
                num_workers=4
            )
        self.model.set_epoch(self.args.attack_epochs)
        self.model.fit(attack_loader)

        # evaluate
        
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
            tmp_neg_set = self.spler.transform(tmp, is_training=False)
            if self.args.problem_type == 'pair':
                tmp_dataset = PairData(tmp_neg_set, is_training=False)
            else:
                tmp_dataset = PointData(tmp_neg_set, is_training=False)
            tmp_loader = torch.utils.data.DataLoader(
                tmp_dataset,
                batch_size=128, 
                shuffle=False, 
                num_workers=0
            )
            topk_indices = self.model.evaluate_model(tmp_loader, self.args.topk)
            preds[u] = torch.take(torch.tensor(test_ucands[u]), topk_indices).cpu().numpy()
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
if __name__=='__main__':
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

    if args.random_method == 'r':
        target_items_results_df.to_csv(f'result/{method}/seed_{args.seed}_epi{episode_length}_att{eval_n_attacker}_dim{args.init_dim}{args.dim1}{args.dim2}_kghop{args.kg_hop}.csv', index=False)
    else:
        target_items_results_df.to_csv(f'result/{method}/seed_{args.seed}_action_size_{args.action_size}_epi{episode_length}_att{eval_n_attacker}_dim{args.init_dim}{args.dim1}{args.dim2}_kghop{args.kg_hop}.csv', index=False)
    # kg neighbor attack
