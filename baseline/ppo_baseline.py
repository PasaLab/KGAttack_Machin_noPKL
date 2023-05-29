import torch
from collections import namedtuple
import torch.optim as optim
import random
import torch.nn as nn
import numpy as np
import itertools
import time

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


class PPOB(object):
    def __init__(self, n_item, boundary_userid, actor_lr, critic_lr, l2_norm, gcn_net, actor, memory_size, eps_start, eps_end, eps_decay,
                 batch_size, gamma, tau=0.01):
        self.actor = actor
        # self.critic = critic
        self.gcn_net = gcn_net
        self.memory = ReplayMemory(memory_size)
        self.global_step = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_item = n_item
        self.batch_size = batch_size
        self.gamma = gamma
        self.start_learning = 10000
        self.tau = tau
        # self.actor_optimizer = optim.Adam(critic.parameters(), lr=actor_lr,
        #                       weight_decay= l2_norm)
        # self.critic_optimizer = optim.Adam(actor.parameters(), lr=critic_lr)
        '''
        todo:
        self.optimizer = optim.Adam(itertools.chain(self.actor1.parameters(),self.actor2.parameters(), self.gcn_net.parameters()), lr=actor_lr, weight_decay = l2_norm)
        '''
        if self.gcn_net is not None:
            self.optimizer = optim.Adam(itertools.chain(self.actor.parameters(),self.gcn_net.parameters()), lr=actor_lr, weight_decay = l2_norm)
        else:
            self.optimizer = optim.Adam(itertools.chain(self.actor.parameters()), lr=actor_lr, weight_decay = l2_norm)
        # self.loss = nn.MSELoss()
        
        self.user_embedding = nn.Embedding(boundary_userid,50)
        self.item_embedding = nn.Embedding(n_item, 50) 
    
    '''to do tonight '''
    def choose_action(self, state, candi=None, is_test = False):
        '''
        todo
        input:
            state: [B*l]
            a1_idx:[B]
            candis based on a1: [B*C]
        intermediate:
            state_emb: [B*(L+1)*E] 或者B*1*E L代表了episode length,
                1代表了state的第一个embedding来自于user_embedding, L代表了剩余的embedding来自于gcn_net的embedding
                gcn_net的embbedding来源可以是初始化look_up_table,size为KG的entity数量，也可以是预训练的embedding
            a1_emb:[B*1*E]
            cat_emb = [state_emb, a1_emb] [B*(L+2)*E]
            candi_emb: [B*C*E] 代表了candidate的embedding,来自于gcn_net
            action_score: [B*C] 通过actor网络求解每一行每一个action的logit
            action_dist: [B*C] 对action_score进行softmax求得prob
            a_idx: [B] 对每一行的action_dist进行采样得到采样的idx
            action: [B] 根据idx找到每一行对应的item
            action_prob: [B] 根据idx找到每一个action对应的prob
        retrun: 
            a2: [B]
            a2_prob: [B]
            a2_idx: [B]
        '''

        '''生成pre state embedding'''
        # u_state_emb = self.user_embedding(torch.LongTensor(state[:,0])).cuda().unsqueeze(1)# [N*1*E]
        if self.gcn_net is None:
            state_emb = self.item_embedding(torch.LongTensor(state[:,1:])).cuda()
        else:
            state_emb = self.gcn_net(state[:,1:]).cuda()# [N*L*E]
        # state_emb = torch.cat((u_state_emb,i_state_emb),dim=1) #[N*(L+1)*E]

        ''' a1 or a2'''
        candi_emb = self.gcn_net.embedding(torch.unsqueeze(torch.LongTensor(candi).cuda(), 0)) # [N*C*E]
        a_logits, a_dist = self.actor(state_emb, candi_emb) # (N,C)
        dist = torch.distributions.Categorical(a_dist) 
        a_idx = dist.sample() # [B]
        action_prob = action_dist[torch.arange(action_dist.shape[0]).type_as(a_idx),a_idx]
        a_idx = a_idx.detach().cpu().numpy()
        
        action = state[list(range(len(state))),a_idx]

        action_prob = action_prob.detach().cpu().numpy()

        return action, action_prob, a_idx
    
    
    def learn(self):
        self.start_learning = 5
        if len(self.memory) < self.start_learning:
            print(f'global step = {self.global_step}, len(self.memory) = {len(self.memory)} ')
            return # debug tag
        
        self.global_step += 1
        transitions = self.memory.get_mem()# batch_size*attack_num=B
        # batch = Transition(*zip(*transitions))
        rewards = 0
        t_loss = 0
        a_t_loss = 0
        step = 1
        for t in transitions: # 这里的t实际上是代表了t，
            step+=1
            # t = Transition(*zip(*t))
                # u_state_emb = self.user_embedding(torch.LongTensor(t.state[:,0])).cuda().unsqueeze(1)# [N*1*E]
            if self.gcn_net is None:
                state_emb = self.item_embedding(torch.LongTensor(t.state[:,1:])).cuda()
            else:
                state_emb = self.gcn_net(t.state[:,1:]).cuda()# [N*L*E]
                # state_emb = torch.cat((u_state_emb,i_state_emb),dim=1) #[N*(L+1)*E]
                # state_emb = torch.unsqueeze(state_emb,0)# [N*(L+1)*E]
            
            # a2
            a_logits, a_dist = self.actor(state_emb, candi_emb)

            a_idx = torch.LongTensor(t.a_idx).cuda()
            a_new_prob = a_dist[torch.arange(a_dist.shape[0]),a_idx] # [B] # 用之前采样的a1_idx新的a1_prob
            a_old_prob = torch.FloatTensor(t.a_prob).cuda()

            a_loss = nn.CrossEntropyLoss(reduction='none')(a_logits, a_idx)
            a_ratio =  a_new_prob / a_old_prob
            a_t_loss += torch.min(a1_ratio, torch.clamp(a_ratio, 0.8, 1.2))*a_loss

            if t.done_mask == 0:
                rewards = t.reward
                rewards = torch.FloatTensor(rewards).cuda()
                
        L = -torch.mean(a_t_loss*rewards)
        print(f'baseline L loss = {L}')
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()