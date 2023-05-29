import torch
import json
import random
import time
import os
import copy
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import sys
sys.path.append('machin')
''' Argument '''
from Args import parse_args 
''' Environment '''
# from Env_kgcn import Env
# print('ENV KGCN')
# from Env_daisy import Env
# print('ENV NeuMF')
from Env_sage import Env
print('ENV SAGE')
'''Net'''
from actor_critic import ActorB, ActorH, ActorP, Critic, CriticP
from gat import GraphEncoder 
'''Agent'''

from machin.frame.algorithms import PPO
from machin.frame.algorithms.utils import safe_call
def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
def run(target_item_idx,args):
    '''args'''
    set_global_seeds(args.seed)
    observe_dim = 4
    action_num = 2
    max_episodes = 1000
    max_steps = 200
    solved_reward = 190
    solved_repeat = 5
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu" )
    '''env'''
    env = Env(args)
    num_users, num_items, num_items_in_kg, target_items, kg, num_indexed_entity, num_indexed_relation,\
            adj_entity, adj_relation, adj_item_dict,_,_ = env.get_meta()
    print('============target_item_embedding===========')
    print(target_items)
    '''net'''
    gnn = GraphEncoder(adj_entity, args).to(args.device)
    print('args.candidate_mode',args.candidate_mode)
    if args.candidate_mode == 'agent_select':
        actor = ActorH(gnn, args).to(args.device)
        critic = Critic(gnn,args).to(args.device)
    elif args.candidate_mode == 'all_item':
        actor = ActorP(num_items, args).to(args.device)
        critic = CriticP(num_items,args).to(args.device)
    else:
        actor = ActorB(gnn, args).to(args.device)
        critic = Critic(gnn,args).to(args.device)
    # actor = ActorB(gnn, args).to(args.device)
    # actor1 = Actor1(ecoding = gnn).to(args.device)
    # actor2 = Actor2(encoding = gnn).to(args.device)
    

    '''agent'''
    ppo = PPO(actor, critic, torch.optim.Adam, torch.nn.MSELoss(reduction="sum"),actor_update_times = args.actor_update, critic_update_times = args.critic_update, discount = args.discount)
    
    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward, total_reward = 0, 0
    
    # train
    # state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
    
    target_item = target_items[target_item_idx]
    print(f'==========target item [{target_item}]=========')
    final_results_train = {'hr':{5:[],10:[],20:[]}, 'ndcg':{5:[],10:[],20:[]}}
    final_results_val = {'hr':{5:[],10:[],20:[]}, 'ndcg':{5:[],10:[],20:[]}}
    pivot_book = {}
    while episode < args.max_episodes: 
        # print(f'==========env reset===========')
        cur_state = env.reset(target_item, episode) # todo: cur_state have the mask
        # print(cur_state)
        r, done = 0, False
        tmp_observations = defaultdict(list)
        for i in range(args.episode_length-1): # -1 is that the first is user id
            # with torch.no_grad():
                action = safe_call(actor, cur_state)[0] # [batch_size*1] or [batch_size*2]
                if args.candidate_mode == 'agent_select':
                    a_idx, pivot_idx = action[0], action[1]
                    # pivot_book[f'{episode}_{i}'] = pivot_idx.cpu().numpy()
                else:
                    a_idx, pivot_idx = action, None
                # print('=======pivot_idx=======')
                # print(pivot_idx)
                new_state,r,done,info = env.step(a_idx, pivot_idx) # action_chosen: cuda Tensor [B], r: [B] ,done: [1]
                
                total_reward += r
                # add to the transition
                for i in range(args.num_train_attacker):
                    # to tensor the cur_state 需要被-1补全
                    # cur_state = torch.tensor(cur_state, dtype=t.float32).view(1, observe_dim)
                    # new_state = torch.tensor(new_state, dtype=t.float32).view(1, observe_dim)
                    if args.candidate_mode == 'agent_pivot':
                        tmp_observations[i].append(
                                {
                                    "state": { key:val[i].unsqueeze(0) for key, val in cur_state.items()}, # unsqueeze since 1st dim should be batch
                                    "action": {"item_action": a_idx[i].unsqueeze(0),'pivot_action':pivot_idx[i].unsqueeze(0) }, # dimension check?
                                    "next_state": { key:val[i].unsqueeze(0) for key, val in new_state.items()}, # unsqueeze since 1st dim should be batch
                                    "reward": r[i], # not tensor
                                    "terminal": done[i], # not tensor
                                }
                            )
                    else:
                        tmp_observations[i].append(
                                {
                                    "state": { key:val[i].unsqueeze(0) for key, val in cur_state.items()}, # unsqueeze since 1st dim should be batch
                                    "action": {"item_action": a_idx[i].unsqueeze(0)}, # dimension check?
                                    "next_state": { key:val[i].unsqueeze(0) for key, val in new_state.items()}, # unsqueeze since 1st dim should be batch
                                    "reward": r[i], # not tensor
                                    "terminal": done[i], # not tensor
                                }
                            )
                cur_state = new_state
        for k in [5, 10, 20]:
            final_results_train['hr'][k].append(info['res_train'].loc[0,k])
            final_results_train['ndcg'][k].append(info['res_train'].loc[1,k])
            final_results_val['hr'][k].append(info['res_val'].loc[0,k])
            final_results_val['ndcg'][k].append(info['res_val'].loc[1,k])
        # print('cur_state',cur_state)
        for i in range(args.num_train_attacker):            
            ppo.store_episode(tmp_observations[i])
        print('update!')
        ppo.update()      
        
        episode +=1
    # pivot_book_df = pd.DataFrame(pivot_book)
    # pivot_book_df.to_csv(f'processed_data_{args.dataset}/pivot_book_{target_item}.csv')
    print(final_results_val['hr'][5], final_results_val['hr'][10], final_results_val['hr'][20])
    return final_results_val['hr'][5], final_results_val['hr'][10], final_results_val['hr'][20],final_results_val['ndcg'][5], final_results_val['ndcg'][10], final_results_val['ndcg'][20]
    

if __name__=='__main__':
    args = parse_args()
    attack_hr_result_5 = defaultdict(list)
    attack_hr_result_10 = defaultdict(list)
    attack_hr_result_20 = defaultdict(list)
    attack_ndcg_result_5 = defaultdict(list)
    attack_ndcg_result_10 = defaultdict(list)
    attack_ndcg_result_20 = defaultdict(list)
    start_time = time.time()
    for item_idx in range(args.num_target_item):
        hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = run(item_idx,args)
        attack_hr_result_5[item_idx] = hr5
        attack_hr_result_10[item_idx] = hr10
        attack_hr_result_20[item_idx] = hr20
        attack_ndcg_result_5[item_idx] = ndcg5
        attack_ndcg_result_10[item_idx] = ndcg10
        attack_ndcg_result_20[item_idx] = ndcg20
    print('total  time = ', time.time()-start_time)
    attack_hr5_result_df = pd.DataFrame(attack_hr_result_5)
    attack_hr5_result_df[-1] = attack_hr5_result_df.mean(1)
    attack_hr10_result_df = pd.DataFrame(attack_hr_result_10)
    attack_hr10_result_df[-1] = attack_hr10_result_df.mean(1)
    attack_hr20_result_df = pd.DataFrame(attack_hr_result_20)
    attack_hr20_result_df[-1] = attack_hr20_result_df.mean(1)
    attack_ndcg5_result_df = pd.DataFrame(attack_ndcg_result_5)
    attack_ndcg5_result_df[-1] = attack_ndcg5_result_df.mean(1)
    attack_ndcg10_result_df = pd.DataFrame(attack_ndcg_result_10)
    attack_ndcg10_result_df[-1] = attack_ndcg10_result_df.mean(1)
    attack_ndcg20_result_df = pd.DataFrame(attack_ndcg_result_20)
    attack_ndcg20_result_df[-1] = attack_ndcg20_result_df.mean(1)
    print(f'=========final mean hr method {args.candidate_mode} action_size {args.action_size} seed {args.seed} bandi_ratio {args.bandit_ratio}======')
    print('hr20')
    print(attack_hr20_result_df[-1])
    print('hr10')
    print(attack_hr10_result_df[-1])
    print('hr5')
    print(attack_hr5_result_df[-1])
    print('ndcg20')
    print(attack_ndcg20_result_df[-1])
    print('ndcg10')
    print(attack_ndcg10_result_df[-1])
    print('ndcg5')
    print(attack_ndcg5_result_df[-1])
    attack_hr20_result_df.to_csv(f"""result/{args.candidate_mode}/seed_{args.seed}
                                _bandit_ratio{args.bandit_ratio}
                                _actino_size{args.action_size}
                                _epi{args.episode_length}
                                _att{args.num_train_attacker}
                                _dim{args.init_dim}{args.dim1}{args.dim2}
                                _kghop{args.kg_hop}
                                _action_size{args.action_size}
                                _topk{args.attack_topk}
                                _actor{args.actor_hidden}    
                                _critic{args.critic_hidden}
                                _gru{args.gru_hidden}{args.gru_layer}
                                _gnn{args.max_aggre_neighbor_entity}{args.gcn_hidden}{args.gcn_layer}.csv""", index=False)