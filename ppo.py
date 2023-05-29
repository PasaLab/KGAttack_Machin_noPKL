import torch
from collections import namedtuple
import torch.optim as optim
import random
import torch.nn as nn
import numpy as np
import itertools
import time

Transition = namedtuple('Transition', ('state','a2', 'a2_prob','a2_idx','a1', 'a1_prob','a1_idx', 'reward','intrisic_reward', 'done_mask', 'candi'))

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

class PPO(object):
    def __init__(self, n_action, boundary_userid, actor_lr, critic_lr, l2_norm, gcn_net, actor1, actor2, critic, memory_size, eps_start, eps_end, eps_decay,
                 batch_size, gamma, tau=0.01):
        self.actor1 = actor1
        self.actor2 = actor2
        # self.critic = critic
        self.gcn_net = gcn_net
        self.memory = ReplayMemory(memory_size)
        self.global_step = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_action = n_action
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
        self.optimizer = optim.Adam(itertools.chain(self.actor1.parameters(),self.actor2.parameters(),self.gcn_net.parameters()), lr=actor_lr, weight_decay = l2_norm)
        # self.loss = nn.MSELoss()
        self.user_embedding = nn.Embedding(boundary_userid,50)

    def choose_action(self, state, candi=None, a1=None, is_test = False): # actor2
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
        if state.shape[1]==1:
            state_emb = self.user_embedding(torch.LongTensor(state)).cuda() # [N*1*E]

        else:
            u_state_emb = self.user_embedding(torch.LongTensor(state[:,0])).cuda().unsqueeze(1)# [N*1*E]
            # s_time = time.time()
            i_state_emb = self.gcn_net(state[:,1:]).cuda()# [N*L*E]
            # print(f'gcn_net time = {time.time()-s_time}')
            state_emb = torch.cat((u_state_emb,i_state_emb),dim=1) #[N*(L+1)*E]
            # print('state_emb[2]', state_emb[2][30])
            # state_emb = torch.unsqueeze(state_emb,0)# [N*(L+1)*E]
        # candi_emb = self.gcn_net.embedding(torch.unsqueeze(torch.LongTensor(candi).cuda(), 0)) # [N*C*E]
        ''' a1 or a2'''
        if candi is None:
            action_score, action_dist = self.actor1(state_emb, state.shape[1])
            dist = torch.distributions.Categorical(action_dist) 
            a_idx = dist.sample() # [B]

            action_prob = action_dist[torch.arange(action_dist.shape[0]).type_as(a_idx),a_idx]
            a_idx = a_idx.cpu().numpy()
            
            action = state[list(range(len(state))),a_idx]

            action = action
            action_prob = action_prob.detach().cpu().numpy()
            return action, action_prob, a_idx
        else:
            action_b = []
            action_prob_b = []
            a_idx_b = []
            # 每一行candi不一样长
            for i in range(state.shape[0]):
                _a1 = a1[i]
                _state_emb = state_emb[i].unsqueeze(0)
                _candi = candi[i]
                _candi = torch.LongTensor(_candi).cuda().unsqueeze(0)
                _candi_emb = self.gcn_net.embedding(_candi) # [B=1*C*E]
                action_score, action_dist = self.actor2(_state_emb, _candi_emb, _a1) # [B=1,C],[B=1,C]
                dist = torch.distributions.Categorical(action_dist) 
                a_idx = dist.sample() # [B=1]
                
                action = _candi[torch.arange(_candi.shape[0]).type_as(a_idx),a_idx] #[B=1]
                action_prob = action_dist[torch.arange(action_dist.shape[0]).type_as(a_idx),a_idx] # [B=1]
                
                action = action.cpu().numpy()
                action_prob = action_prob.detach().cpu().numpy()
                a_idx = a_idx.cpu().numpy()

                action_b.extend(action)
                action_prob_b.extend(action_prob)
                a_idx_b.extend(a_idx)
            return action_b, action_prob_b, a_idx_b

            
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.global_step * self.eps_decay)
        # if is_test or random.random() > eps_threshold:
        #     action_dist = self.actor(state_emb, candi_emb)
        #     dist = torch.distributions.Categorical(action_dist)
        #     a_idx = dist.sample().item()
        #     action = candi[a_idx]
        #     action_prob = action_dist[0][a_idx].item()
        # else:
        #     action = random.randrange(self.n_action)
        #     action_prob = 1/self.n_action

        
        

    
    '''
    todo:
    input:
        state: [B*l]
        candi: [B*C]
    intermediate:
        state_emb: [B*(L+1)*E] 或者B*1*E L代表了episode length,
            1代表了state的第一个embedding来自于user_embedding, L代表了剩余的embedding来自于gcn_net的embedding
        action_score: [B*C] 通过actor网络求解每一行每一个action的logit
        action_dist: [B*C] 对action_score进行softmax求得prob
        a_idx: [B] 对每一行的action_dist进行采样得到采样的idx
        action: [B] 根据idx找到每一行对应的item
        action_prob: [B] 根据idx找到每一个action对应的prob
    output:
        a1:[B]
        a1_prob:[B]
        a1_idx:[B]

    '''
    

        
    def learn(self):
        '''
        input: 
            None
        intermediate:
            计算公式7中的Loss Fucntion，并利用torch的auto—differentiation来求
            1. 利用torch.CrossEntropyLoss来求解 \delta \pi(a_t|s), 即把a_idx看成是ground truth，然后通过CrossEntropyLoss即可求出log概率 [B]
            2. action_dist在每一轮梯度更新之后会变化，ratio=old_prob/new_prob, old_prob是最初inference时候的prob，new_prob是actor网络参数更新之后的prob，都是a_idx下的
            3. t_loss+=min(ratio, clip(ratio) )*loss，求出episode length长度下的总的loss
            4. 和reward相乘求得总的loss L
            5. 对L反向梯度求解更新actor参数
        output:
            None

        '''
        self.start_learning = 5
        if len(self.memory) < self.start_learning:
            print(f'global step = {self.global_step}, len(self.memory) = {len(self.memory)} ')
            return # debug tag
        
        self.global_step += 1
        transitions = self.memory.get_mem()# batch_size*attack_num=B
        # batch = Transition(*zip(*transitions))
        rewards = 0
        t_loss = 0
        a1_t_loss = 0
        a2_t_loss = 0
        step = 1
        for t in transitions: # 这里的t实际上是代表了t，
            step+=1
            # t = Transition(*zip(*t))
            if t.state.shape[1]==1:
                state_emb = self.user_embedding(torch.LongTensor(t.state)).cuda() # [N*1*E]

            else:
                u_state_emb = self.user_embedding(torch.LongTensor(t.state[:,0])).cuda().unsqueeze(1)# [N*1*E]
                i_state_emb = self.gcn_net(t.state[:,1:]).cuda()# [N*L*E]
                state_emb = torch.cat((u_state_emb,i_state_emb),dim=1) #[N*(L+1)*E]
                # state_emb = torch.unsqueeze(state_emb,0)# [N*(L+1)*E]
            # candi_emb = self.gcn_net.embedding(torch.unsqueeze(torch.LongTensor(candi).cuda(), 0)) # [N*C*E]
            
            # action_score, action_dist = self.actor(state_emb, candi_emb) # [B,C],[B,C] 对candi重新做一次计算，得到新的score和新的action_prob，新是因为后面多次训练迭代的时候，神经网路参数会更新，因此每一次forward的时候得到的结果是新的（在没有gradient update的时候得到的结果和transition里的结果是一样的）
            

            # a1
            a1_score, a1_dist = self.actor1(state_emb, t.state.shape[1])

            a1_idx = torch.LongTensor(t.a1_idx).cuda()
            a1_new_prob = a1_dist[torch.arange(a1_dist.shape[0]),a1_idx] # [B] # 用之前采样的a1_idx新的a1_prob
            a1_old_prob = torch.FloatTensor(t.a1_prob).cuda()

            a1_loss = nn.CrossEntropyLoss(reduction='none')(a1_score, a1_idx)
            a1_ratio =  a1_new_prob / a1_old_prob
            a1_t_loss += torch.min(a1_ratio, torch.clamp(a1_ratio, 0.9, 1.1))*a1_loss

            
            # a2
            a2_new_prob = []
            a2_loss = []
            # 每一行candi不一样长
            for i in range(t.state.shape[0]):
                _a2_idx = torch.tensor(t.a2_idx[i]).unsqueeze(0).cuda()
                _a1_idx = a1_idx[i]
                _state_emb = state_emb[i].unsqueeze(0)
                _candi = t.candi[i]
                _candi = torch.LongTensor(_candi).cuda().unsqueeze(0)
                _candi_emb = self.gcn_net.embedding(_candi) # [B=1*C*E]
                _a2_score, _a2_dist = self.actor2(_state_emb, _candi_emb, _a1_idx) # [B=1,C],[B=1,C]
                
                # action = _candi[torch.arange(_candi.shape[0]).type_as(a2_idx),a2_idx] #[B=1]
                _a2_new_prob = _a2_dist[torch.arange(_a2_dist.shape[0]), _a2_idx] # [B=1]
                _a2_loss = nn.CrossEntropyLoss(reduction='none')(_a2_score, _a2_idx)
                a2_loss.append(_a2_loss)
                a2_new_prob.extend(_a2_new_prob)
            # a2_score, a2_dist = self.actor2(state_emb, candi_emb) # candi_emb 是a1产生的

            # a2_idx = torch.LongTensor(t.a2_idx).cuda()
            # a2_new_prob = a2_dist[torch.arange(a2_dist.shape[0]).type_as(a2_idx),a2_idx] # [B] # 用之前采样的a1_idx新的a1_prob
            a2_Old_Prob = torch.FloatTensor(t.a2_prob).cuda()

            # a2_loss = nn.CrossEntropyLoss(reduction='none')(a2_score, a2_idx) # debug?
            a2_L = torch.stack(a2_loss)
            a2_New_Prob = torch.stack(a2_new_prob)
            a2_ratio =  a2_New_Prob / a2_Old_Prob # debug new / old?
            a2_t_loss += torch.min(a2_ratio, torch.clamp(a2_ratio, 0.9, 1.1))*a2_L


            # if t.done_mask == 0:
            #     rewards = t.reward
            #     rewards = torch.FloatTensor(rewards).cuda()
            
            # # if step%31 ==0 :
            # #     print('action_score',action_score)
            # #     print('action_dist',action_dist)
            # # action = candi[torch.arange(candi.shape[0]).type_as(a_idx),a_idx] #[B]
            # a_idx = torch.LongTensor(t.a_idx).cuda() # 之前采样的样本
            # new_prob = action_dist[torch.arange(action_dist.shape[0]).type_as(a_idx),a_idx] # [B] 用之前采样的样本idx获得此刻的action_prob
            # old_prob = torch.FloatTensor(t.prob).cuda()
            # loss = nn.CrossEntropyLoss(reduction='none')(action_score, a_idx) # [B]
            # ratio = old_prob / new_prob
            # t_loss += torch.min(ratio, torch.clamp(ratio, 0.9, 1.1))*loss
            # loss = action_score 交叉熵 one-hot(t.action) # 用老的action和当前的action score做交叉熵
            # ratio = old_prob/new_prob
            # clip_ratio = min(ratio, clip(ratio))
            # t_loss += loss*clip_ratio
            if t.done_mask == 0:
                rewards = t.reward
                intrisic_reward = t.intrisic_reward
                rewards = torch.FloatTensor(rewards).cuda()
                intrisic_reward = torch.FloatTensor(intrisic_reward).cuda()
                
        L1 = -torch.mean(a1_t_loss*intrisic_reward) # 
        L2 = -torch.mean(a2_t_loss*rewards)
        print(f'L1 loss = {L1}, L2 loss = {L2}')
        self.optimizer.zero_grad()
        L1.backward(retain_graph=True)
        L2.backward()
        self.optimizer.step()
        
        # L = torch.mean(t_loss*rewards)
        # self.optimizer.zero_grad()
        # L.backward()
        # self.optimizer.step()



        # b_s = self.gcn_net(list(batch.state))       #[B*max_node*emb_dim]
        # b_a = torch.LongTensor(np.array(batch.action).reshape(-1, 1)).cuda()  #[B*max_node]
        # b_a_emb =self.gcn_net.embedding(b_a)       #[B*max_node*emb_dim]
        # b_p = torch.stack(batch.prob)#[B*max_node]
        
        # b_r = torch.FloatTensor(np.array(batch.reward).reshape(-1, 1)).squeeze(1).cuda()#[B]
        # b_msk = torch.Tensor(np.array(batch.done_mask).reshape(-1, 1)).squeeze(1).cuda()#[B*max_node]
        # next_candi = torch.LongTensor(list(batch.next_candi)).cuda() #[N*k]
        # next_candi_emb = self.gcn_net.embedding(next_candi)    #[N*k*emb_dim]

        # # calculate u_t(REINFORCE: using a trajectory, compute the discounted return u_t to approximate Q_pi(s,a))
        # # This part can be removed to the Memory class
        # returns = torch.zeros_like(b_r)#[N*1]
        # running_returns = 0
        # for t in reversed(range(0, len(transitions))):
        #     running_returns = b_r[t] + self.gamma * running_returns * b_msk[t]
        #     returns[t] = running_returns

        # returns = (returns - returns.mean()) / returns.std() # [N]
        
        # # REINFORCE with baseline
        
        # # train critic
        # values = self.critic(b_s).squeeze(1) #[N]?
        # loss = self.loss(values, returns)
        # self.critic_optimizer.zero_grad()
        
        # # train actor      
        # log_prob = torch.log(b_p)
        # objective = returns * log_prob
        # objective = - objective.mean()
        # self.actor_optimizer.zero_grad()
        
        # loss.backward(retain_graph=True)
        # objective.backward()
        # self.critic_optimizer.step()
        # self.actor_optimizer.step()
