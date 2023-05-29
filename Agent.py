import torch
import json
import random
import time
import os
import copy
import numpy as np
from tqdm import tqdm
from Args import parse_args

from Env_sage import Env
from actor_critic import Actor1, Actor2, Critic
from ppo import PPO
# from gcn import GraphEncoder
from gat import GraphEncoder


def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)

class Agent(object):
    def __init__(self, train_n_attacker, episode_length, dataset_name, target_item_idx, l2_norm = 1e-3, alpha = 200):

        set_global_seeds(28)
        self.alpha = alpha
        self.target_item_idx = target_item_idx

        self.eval_candi_num = 100
        self.max_candi_num =  1000
        self.attack_topk = 20
        self.attack_epoch = 10
        self.target_item_num = 10
        self.n_max_attacker = 1000
        self.eval_n_attacker = 10
        self.train_n_attacker = train_n_attacker
        self.eval_spy_users_num = 500
        self.train_spy_users_num = 50
        self.dataset_name= dataset_name 
        self.processed_path = 'processed_data_'+self.dataset_name
        self.metric = 'hr' # mdcg
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
        self.l2_norm = l2_norm

        self.sample_times = 30
        self.update_times = 3


        self.actor1 = Actor1(emb_size=emb_size).cuda()
        self.actor2 = Actor2(emb_size=emb_size).cuda()
        self.critic = Critic(emb_size=emb_size).cuda()

        # self.gcn = GraphEncoder(n_entity, emb_size, embeddings=embeddings, max_seq_length=32, max_node=40, hiddim = 50, layers=self.gcn_layer,
        #                         cash_fn='data/kg/'+KG_VEC[self.dataset_name]+'/cache_graph-40.pkl.gz',fix_emb=self.fix_emb, adj_entity = self.adj_entity).cuda()
        self.gcn = GraphEncoder(adj_entity, args).cuda()
        self.ppo = PPO(self.num_items_in_kg, self.boundary_userid, self.learning_rate, self.learning_rate, self.l2_norm, self.gcn, self.actor1, self.actor2, self.critic, self.memory_size, self.eps_start, self.eps_end, self.eps_decay, self.batch_size,
                  self.gamma, self.tau)


    def attack_candidate(self,cur_state, mask, a1_idx, target_item):
        if cur_state.shape[1] == 1:
            candis = []
            for i in range(cur_state.shape[0]):
                candis.append([target_item])
            return np.array(candis)
        else:
            candis = []
            # mask = np.array(mask).T
            for b_idx in range(len(cur_state)):
                # m = mask[b_idx]
                if a1_idx[b_idx] == 0:  # 如果选择的是user
                    # tmp = set()
                    # if not tmp:
                    #     candi = random.sample(list(range(self.num_items_in_kg)), len(self.target_items)*self.alpha)
                    #     candis.append(candi)
                    # else:
                    #     candis.append(list(tmp))
                    # tmp = set(self.adj_item_dict[target_item]) - set(m)
                    tmp = set(self.adj_item_dict[target_item]) 
                    if len(tmp)< self.alpha:
                        rr = self.alpha - len(tmp)
                        candi = random.sample(list(range(self.num_items_in_kg)), rr) + list(tmp)
                        candis.append(candi)
                    else:
                        candis.append(random.sample(list(tmp), self.alpha))
                else: #选择了某一个item
                    item = cur_state[b_idx][a1_idx[b_idx]]
                    # tmp = set(self.adj_item_dict[item])-set(m)
                    tmp = set(self.adj_item_dict[item])
                    # if not tmp:
                    #     candi = random.sample(list(range(self.num_items_in_kg)), len(self.target_items)*self.alpha)
                    #     candis.append(candi)
                    # else:
                    #     candi = list(tmp)  + random.sample(list(range(self.num_items_in_kg)), len(self.target_items)*self.alpha)
                    #     candis.append(candi )
                    if len(tmp)< self.alpha:
                        rr = self.alpha - len(tmp)
                        candi = random.sample(list(range(self.num_items_in_kg))+self.adj_item_dict[target_item], rr) + list(tmp)
                        candis.append(candi)
                    else:
                        # candis.append(list(tmp) + random.sample(list(range(self.num_items_in_kg)), len(self.target_items)*alpha) )
                        candis.append(random.sample(list(tmp), self.alpha))
                # print(f'len (candi) , {len(candi)}')
            return np.array(candis)
    
    def train(self):
        
        # for target_item in self.target_items:
        target_item = self.target_items[self.target_item_idx]
        r_nb = {'eval_hr':[],'eval_ndcg':[],'hr':[],'ndcg':[]}
        target_nb = {}
        max_hr = -1
        max_ndcg = -1
        from collections import defaultdict
        pivot_list = defaultdict(list)
        for itr in tqdm(range(self.sample_times),desc='sampling'):
            r, done  =  0, False
            eval_hr, eval_ndcg = -1, -1
            # 改成新生成user_id?
            # user_id = [i for i in range(self.num_users+itr*self.train_n_attacker, self.num_users+(itr+1)*self.train_n_attacker)]
            print('=====user_id=====')
            print(user_id)
            cur_state = self.env.attack_reset(target_item)
            candidate_mask = []
            while not done: # a trajectory (s_0,a_0,r_0,...,s_t,a_t,r_t)
                a1, a1_prob, a1_idx = self.ppo.choose_action(cur_state)
                print('=====pivot=====')
                # print(a1_idx)
                # pivot_list[itr].append(a1_idx) 
                candi = self.attack_candidate(cur_state, candidate_mask, a1_idx, target_item)
                a2, a2_prob, a2_idx = self.ppo.choose_action(cur_state, candi, a1_idx)

                new_state,r,intrisic_r, done,hr,ndcg,eval_hr, eval_ndcg = self.env.attack_step(a1 = a1_idx, action_chosen = a2, target_item= target_item, metric = self.metric) # action_chosen: cuda Tensor [B], r: [B] ,done: [1]
                done_mask = 0 if done else 1 # 到sub budget的时候就要返回done
                # candidate_mask.append(a2) # 记录了profile里的item，没有记录profile中的第一个，因为第一个是user_id
                
                if cur_state.shape[1] !=0:
                    self.ppo.memory.push(cur_state, a2, a2_prob, a2_idx, a1, a1_prob, a1_idx, r, intrisic_r, done_mask, candi)
                
                cur_state = new_state
            for _itr in tqdm(range(self.update_times),desc='updating'): 
                self.ppo.learn()
            self.ppo.memory.reset()
            # hr, ndcg = self.attack_evaluate(target_item) # eval!!
            r_nb['eval_hr'].extend([eval_hr])
            r_nb['eval_ndcg'].extend([eval_ndcg])
            r_nb['hr'].extend([hr])
            r_nb['ndcg'].extend([ndcg])
            max_hr = max(r_nb['hr'])
            max_ndcg = max(r_nb['ndcg'])
            
            # intermediate result
            # if itr % 5 ==0:
            #     with open(self.processed_path+'/intermediate_promotion_result.txt','a+') as f:
            #         f.write(f'target_item: {target_item}, evaluate result: {r_nb} topk {self.attack_topk} the max hr is {max_hr},the max ndcg is {max_ndcg} under eval spy user num {self.eval_spy_users_num}, n_attacker per train {self.train_n_attacker}, n_attacker per eval {self.eval_n_attacker}')
            #         f.write('\n')
        # total result
        import pandas as pd
        pd.DataFrame(pivot_list).to_csv('./pivot_count0_softmax.csv')
        print(f'EVALUATE reward dict: {r_nb}, the max hr is {max_hr}, the max ndcg is {max_ndcg}' )
        # with open(self.processed_path+f'/promotion_eval_result_{self.attack_topk}.txt','a+') as f:
        #     f.write(f'target_item: {target_item}, evaluate result: {r_nb} topk {self.attack_topk} the max hr is {max_hr},the max ndcg is {max_ndcg} under eval spy user num {self.eval_spy_users_num}, n_attacker per train {self.train_n_attacker}, n_attacker per eval {self.eval_n_attacker}')
        #     f.write('\n')
        # with open(self.processed_path+f'/promotion_train_result_{self.attack_topk}.txt','a+') as f:
        #     f.write(f'target_item: {target_item}, evaluate result: {r_nb} topk {self.attack_topk} under eval spy user num {self.eval_spy_users_num}, n_attacker per train {self.train_n_attacker}, n_attacker per eval {self.eval_n_attacker}')
        #     f.write('\n')
        # candi num 50 100 200 400 800 1600
        with open(self.processed_path+f'/promotion_train_result_{self.attack_topk}_{self.alpha}.txt','a+') as f:
            f.write(f'target_item: {target_item}, evaluate result: {r_nb} topk {self.attack_topk} under eval spy user num {self.eval_spy_users_num}, n_attacker per train {self.train_n_attacker}, n_attacker per eval {self.eval_n_attacker}')
            f.write('\n')
    
    def attack_evaluate(self, target_item):
        '''每次评估都是重置Recsys模型'''
        print('Start evaluate')
        r, done  =  0, False
        # todo: user_id 需要和train时候的保持一致
        user_id = [i for i in range(self.num_users, self.num_users+self.eval_n_attacker)] # 
        cur_state = self.env.attack_reset(user_id) # todo: [B=attack_user_num*batch_size] user_id应该比user_count大 
        candidate_mask = []
        while not done:
            a1, a1_prob, a1_idx = self.ppo.choose_action(cur_state)
            candi = self.attack_candidate(cur_state, candidate_mask, a1_idx, target_item)
            a2, a2_prob, a2_idx = self.ppo.choose_action(cur_state, candi, a1_idx)
            new_state,r,done,hr,ndcg, _, _ = self.env.attack_step(action_chosen = a2, target_item=target_item, metric = self.metric) # evaluate时的pretend user数量要修改成500个，和ppo train时的user不一样
            done_mask = 0 if done else 1
            candidate_mask.append(a2)
            cur_state = new_state
        return hr, ndcg
    
    def run(self):
        self.train()
        # if i % self.log_step == 0:
        #     self.attack_evaluate()
        # if i !=0 and i % self.save_step == 0:
        #     self.save_attack_model(i)
        
if __name__=='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # target_item_num = 1

    # train_n_attacker = 10
    # episode_length = 16
    # dataset_name = 'Book-Crossing' # Book-Crossing
    # rec = Agent(train_n_attacker, episode_length, dataset_name, 0, l2_norm = 1e-4, update_times = 3)
    # rec.run()
    # del rec
    
    # train_n_attacker = 10
    # episode_length = 16
    # dataset_name = 'Book-Crossing' # Book-Crossing
    # rec = Agent(train_n_attacker, episode_length, dataset_name, 0, l2_norm = 1e-2, update_times = 3)
    # rec.run()
    # del rec

    # train_n_attacker = 10
    # episode_length = 16
    # dataset_name = 'Book-Crossing' # Book-Crossing
    # rec = Agent(train_n_attacker, episode_length, dataset_name, 0, l2_norm = 1e-3, update_times = 1) # 不行
    # rec.run()
    # del rec

    # train_n_attacker = 10
    # episode_length = 16
    # dataset_name = 'Book-Crossing' # Book-Crossing
    # rec = Agent(train_n_attacker, episode_length, dataset_name, 0, l2_norm = 1e-3, update_times = 2)
    # rec.run()
    # del rec

    # train_n_attacker = 10
    # episode_length = 16
    # dataset_name = 'Book-Crossing' # Book-Crossing
    # rec = Agent(train_n_attacker, episode_length, dataset_name, 0, l2_norm = 1e-3, update_times = 5)
    # rec.run()
    # del rec



    # train_n_attacker = 15
    # episode_length = 16
    # rec = Agent(train_n_attacker, episode_length)
    # rec.run()
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    target_item_num = 10
    for i in range(target_item_num):
        train_n_attacker = 3
        episode_length = 16
        dataset_name = 'ml-1m' # Book-Crossingk
        rec = Agent(train_n_attacker, episode_length, dataset_name, i, alpha = 40)
        rec.run()
        del rec
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')
    # train_n_attacker = 25
    # episode_length = 16
    # rec = Agent(train_n_attacker, episode_length)
    # rec.run()
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')
    # train_n_attacker = 30
    # episode_length = 16
    # rec = Agent(train_n_attacker, episode_length)
    # rec.run()
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')
    # train_n_attacker = 10
    # episode_length = 4
    # rec = Agent(train_n_attacker, episode_length)
    # rec.run()
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')
    # train_n_attacker = 10
    # episode_length = 8
    # rec = Agent(train_n_attacker, episode_length)
    # rec.run()
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')
    # train_n_attacker = 10
    # episode_length = 32
    # rec = Agent(train_n_attacker, episode_length)
    # rec.run()
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')
    # train_n_attacker = 20
    # episode_length = 32
    # rec = Agent(train_n_attacker, episode_length)
    # rec.run()
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')
    # train_n_attacker = 10
    # episode_length = 64
    # rec = Agent(train_n_attacker, episode_length)
    # rec.run()
    # print(f'NEW HYPERPARAMETER:{train_n_attacker}, {episode_length}')