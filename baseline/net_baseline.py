import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorB(nn.Module):
    def __init__(self, emb_size, x_dim = 50, state_dim=50, hidden_dim=50, layer_num=1):
        super(ActorB, self).__init__()
        # candi_num = 200
        # emb_size = 50
        # x_dim = 50
        # self.candi_num = candi_num
        self.rnn = nn.GRU(x_dim,state_dim,layer_num,batch_first=True)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim+emb_size, hidden_dim)   #hidden_dim + emb_size
        self.fc3 = nn.Linear(hidden_dim,1)
    
    def forward(self, x, y):
        """
        :param x: encode state history [N*L*D]; y: action embedding [N*K*E],
            
            N: batch size, L: seq length, D: embedding size, K: action set length
        :return: action score [N*K]
        """
        
        out, h = self.rnn(x)
        h = h.permute(1,0,2) #[N*1*D]
        # test = x[torch.arange(x.shape[0]),a1,:].unsqueeze(1)
        # print(x[0,a1[0],:])
        # print(test[0])
        x = F.relu(self.fc1(h))
        x = x.repeat(1,y.shape[1],1) # [N*K*D]
        state_cat_action = torch.cat((x,y),dim=2)
        action_logits = self.fc3(F.relu(self.fc2(state_cat_action))).squeeze(dim=2) #[N*K]
    
        return action_logits
    
    def get_action(self, x, y):
        a_prob = F.softmax(self.forward(x, y), dim=1)
        dist = torch.distributions.Categorical(a_prob) 
        a_idx = dist.sample()
        return a_idx, a_prob
    
    def get_logprob_entropy(self, state, a_idx):
        a_prob = F.softmax(self.forward(x, y), dim=1)
        dist = torch.distributions.Categorical(a_prob)
        a_int = a_idx.squeeze(1).long()
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_idx, a_prob):
        dist = torch.distributions.Categorical(a_prob)
        return dist.log_prob(action.long().squeeze(1))
    

        
        
    