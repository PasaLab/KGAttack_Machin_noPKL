import torch
import json
import random
import time
import os
import copy
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/root/cjf/KGAttack/')
from Env_sage import Env
from baseline.net_baseline import ActorB
from baseline.ppo_baseline import PPOB
from gat import GraphEncoder

from collections import namedtuple
import torch.optim as optim
import random
import torch.nn as nn
import numpy as np
import itertools
import time
import pandas as pd
from Args import parse_args

Transition = namedtuple('Transition', ('state','a', 'a_prob','a_idx', 'reward', 'done_mask', 'candi'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # def sample(self, batch_size):
    #     return random.sample(self.memory, batch_size)
    def get_mem(self):
        return self.memory

    def reset(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)

class Agent(object):
    def __init__(self, train_n_attacker, episode_length, dataset_name, target_item_idx,alpha=200):

        set_global_seeds(28)

        # baseline_model:
        self.alpha=alpha
        self.prev_neighbor = False
        self.cumu_neighbor = False
        self.all_item = True
        self.all_item_noGNN = False # 改ppo_baseline.py里的 self.gcn
        if self.prev_neighbor:
            self.baseline_method = 'baseline_prev'
        elif self.cumu_neighbor:
            self.baseline_method = 'baseline_cumu'
        elif self.all_item:
            self.baseline_method = 'baseline_all_item'
        elif self.all_item_noGNN:
            self.baseline_method = 'baseline_all_item_noGNN'

        self.target_item_idx = target_item_idx

        # self.eval_candi_num = 100
        # self.max_candi_num =  1000
        # self.attack_topk = 20
        # self.attack_epoch = 10
        # self.target_item_num = 10
        self.n_max_attacker = 1000
        self.eval_n_attacker = 20
        self.train_n_attacker = train_n_attacker
        self.episode_length = episode_length
        self.eval_spy_users_num = 500
        self.train_spy_users_num = 50
        self.dataset_name= dataset_name 
        self.processed_path = 'processed_data_'+self.dataset_name
        self.metric = 'hr' # mdcg
        ''' environment '''
        args = parse_args()
        self.env = Env(args)
        self.num_users, self.num_items, self.num_items_in_kg, self.target_items, self.kg, self.num_indexed_entity, self.num_indexed_relation,\
                self.adj_entity, self.adj_relation, self.adj_item_dict = self.env.get_meta()
        

        self.boundary_userid = int(self.num_users+self.n_max_attacker)

        self.fix_emb = False
        self.gcn_layer = 2
        self.batch_size = 2000
        self.memory_size = 300000
        self.eps_start, self.eps_end, self.eps_decay, self.gamma, self.tau = 0.9, 0.1, 0.0001, 0.7, 0.01
        self.learning_rate = 5e-3
        self.l2_norm = 1e-3

        self.sample_times = 15
        self.update_times = 3

        # KG embedding loading
        KG_VEC = {'ml-1m':'ml1m-kg1m','ml-20m':'ml20-kg500k','Book-Crossing':'bx-kg150k'}
        # embedding_path = 'data/kg/'+KG_VEC[self.dataset_name]+'/embedding.vec.json'
        # embeddings = torch.FloatTensor(json.load(open(embedding_path, 'r'))['ent_embeddings'])
        embeddings = torch.load(f'data/kg/{KG_VEC[self.dataset_name]}/ent_embedding.pt')
        print("load embedding complete!")
        n_entity = embeddings.shape[0] # {movie-lens 1M: 182011}
        emb_size = embeddings.shape[1] # {movie-lens 1M: 50}
        '''net&optimizer: actor, critic, gcn'''
        self.actor = ActorB(emb_size=emb_size).cuda()
        
        if self.all_item_noGNN:
            self.gcn = None
            self.item_embedding = nn.Embedding(num_items, 50)
        else:
            self.gcn = GraphEncoder(n_entity = n_entity,emb_size = emb_size, max_node=40, max_seq_length=129, cached_graph_file=self.processed_path,embeddings=embeddings,\
                                    fix_emb=self.fix_emb, adj_entity = self.adj_entity, hiddim = 50, layers=self.gcn_layer).cuda()

        self.ppo = PPOB(self.num_items_in_kg, self.boundary_userid, self.learning_rate, self.learning_rate, self.l2_norm, self.gcn, self.actor, self.memory_size, self.eps_start, self.eps_end, self.eps_decay, self.batch_size,
                  self.gamma, self.tau)

    def candidate(self,cur_state, mask):
        alpha_for_neighbor = 10
        if self.prev_neighbor:
            candis = []
            mask = np.array(mask).T
            for b_idx in range(len(cur_state)):
                m = mask[b_idx]
                item = cur_state[b_idx][-1] # prev item
                tmp = set(self.adj_item_dict[item])-set(m)
                while len(tmp) == 0:
                    tmp = set(self.adj_item_dict[random.choice(list(range(self.num_items_in_kg)))])-set(m)
                # if len(tmp) == 0:
                #     rr = alpha - len(tmp)
                #     candi = random.sample(list(range(self.num_items_in_kg)), rr)
                #     candis.append(candi)
                if len(tmp)<= self.alpha:
                    # rr = alpha - len(tmp)
                    # candi = random.sample(list(range(self.num_items_in_kg)), rr) + list(tmp)
                    candis.append(list(tmp))
                else:
                    candis.append(random.sample(list(tmp), self.alpha))
            return np.array(candis)
        elif self.cumu_neighbor:
            candis = []
            mask = np.array(mask).T
            for b_idx in range(len(cur_state)):
                m = mask[b_idx]
                tmp = []
                for i in cur_state[b_idx][1:]:
                    tmp+=self.adj_item_dict[i]
                tmp = set(tmp) - set(m)
                if not tmp:
                    candi = random.sample(list(range(self.num_items_in_kg)), len(self.target_items)*self.alpha)
                    candis.append(candi)
                elif len(tmp) < len(self.target_items)*10: 
                    candis.append(list(tmp))
                else:
                    candi = random.sample(list(tmp), len(self.target_items)*self.alpha)
                    candis.append(list(candi))
            return np.array(candis)
        elif self.all_item or self.all_item_noGNN:
            candis = []
            # mask = np.array(mask).T
            for b_idx in range(len(cur_state)):
                # m = mask[b_idx]
                candi = random.sample(list(range(self.num_items_in_kg)), self.alpha) 
                # candi = set(tmp) - set(m)
                candis.append(list(candi))
            return np.array(candis)
    
    def select_action(self, state, candi):
        if self.gcn_net is None:
            state_emb = self.item_embedding(torch.LongTensor(state[:,1:])).cuda()
        else:
            state_emb = self.gcn_net(state[:,1:]).cuda()# [N*L*E]
        candi_emb = self.gcn_net.embedding(torch.unsqueeze(torch.LongTensor(candi).cuda(), 0))
        print('======candi_emb=====')
        print(candi_emb.shape)
        a_idx, probs = self.act.get_action(state_emb, candi_emb)
        return a_idx.detach().cpu().numpy, probs.detach().cpu().numpy()
        
        
    def explore_env(self):
        # for target_item in self.target_items:
        target_item = self.target_items[self.target_item_idx]
        r_nb = {'eval_hr':[],'eval_ndcg':[],'hr':[],'ndcg':[]}

        r, done  =  0, False
        cur_state = self.env.attack_reset(target_item)
        print('=====cur_state=====')
        print(cur_state)
        candidate_mask = []

        while not done: # a trajectory (s_0,a_0,r_0,...,s_t,a_t,r_t)
            candi = self.candidate(cur_state, candidate_mask)
            print('======candi======')
            print(candi.shape)
            a_idx, a_prob = self.select_action(cur_state, candi)
            new_state,r,done,hr,ndcg,eval_hr, eval_ndcg = self.env.attack_step(action_chosen = a_idx, metric = self.metric) # action_chosen: cuda Tensor [B], r: [B] ,done: [1]
            print(new_state)
            
            # done_mask = 0 if done else 1 
            candidate_mask.append(new_state[:,-1]) 
            
            cur_state = new_state

        r_nb['eval_hr'].extend([eval_hr])
        r_nb['eval_ndcg'].extend([eval_ndcg])
        r_nb['hr'].extend([hr])
        r_nb['ndcg'].extend([ndcg])
        
        return r_nb
    def run(self):
        r_nb = self.explore_env()

    
    
        
    
    def attack_evaluate(self, target_item):
        '''每次评估都是重置Recsys模型'''
        print('Start evaluate')
        r, done  =  0, False
        # todo: user_id 需要和train时候的保持一致
        user_id = [i for i in range(self.num_users, self.num_users+self.eval_n_attacker)] # 
        cur_state = self.env.attack_reset(user_id) # todo: [B=attack_user_num*batch_size] user_id应该比user_count大 
        candidate_mask = []
        while not done:
            candi = self.attack_candidate(cur_state, candidate_mask, target_item)
            if self.all_item_noGNN:
                a, a_prob, a_idx = self.ppo.choose_action(cur_state, candi)
            else:
                a, a_prob, a_idx = self.ppo.choose_action(cur_state, candi)
            new_state,r,done,hr,ndcg = self.env.attack_step(action_chosen = a, target_item=target_item, evaluate = True, metric = self.metric) # evaluate时的pretend user数量要修改成500个，和ppo train时的user不一样
            done_mask = 0 if done else 1
            candidate_mask.append(a)
            cur_state = new_state
        return hr, ndcg

    
    
    
if __name__=='__main__':
    # train_n_attacker = 15
    # episode_length = 16
    # rec = Agent(train_n_attacker, episode_length)
    # rec.run()
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    target_item_num = 10
    
    for i in range(target_item_num):
        train_n_attacker = 10
        episode_length = 16
        dataset_name = 'ml-1m' # Book-Crossing
        rec = Agent(train_n_attacker, episode_length, dataset_name, i, alpha = 400)
        rec.run()
        del rec
    
    
