import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
import os

class GraphConstructer():
    def __init__(self, adj_entity, max_nodes, max_seq_length, cached_graph_file='./cached_graph.pkl', args=None):
        self.args = args
        self.max_nodes = max_nodes
        self.max_seq_length = max_seq_length
        self.adj_entity = adj_entity
        assert(self.max_nodes>=len(self.adj_entity[0]))
        self.cached_graph_file = f'{cached_graph_file}/cached_graph_{max_nodes}.pkl'
        if os.path.exists(self.cached_graph_file):
            with open(self.cached_graph_file,'rb') as fin:
                self.cached_graph = pickle.load(fin)
            
        else:
            self.cached_graph = self.get_item_graph()
            with open(self.cached_graph_file,'wb') as fin:
                pickle.dump(self.cached_graph, fin)
    
    def get_item_graph(self):
        print(f'Start getting entity graph by adj_entity')
        E = self.max_nodes
        # 1. get the E*E adj of the node
        adjs = {}
        map = {}
        # print('adj_entity的类型', type(self.adj_entity)) # list
        # print('adj_entity的类型',type(self.adj_entity[0])) # list
        # print('adj_entity的类型',self.adj_entity[0]) #[26132, 5785, 5785, 9489, 12211, 49168, 13082, 8349, 2368, 2437, 6725, 50440, 3893, 2526, \
        # # 49168, 9207, 15091, 44275, 49168, 9207, 9937, 50573, 50440, 53639, 49168, 23675, 6725, 43007, 13731, 26132, 9489, 13082]
        for entity in range(len(self.adj_entity)):
            # 1. entity_id to idx
            id2idx = {}
            id2idx[entity] = 0
            idx = 1
            for id in self.adj_entity[entity]:
                if not id in id2idx:# 避免neighbors中的重复item
                    id2idx[id] =idx
                    idx+=1
                if idx == E:
                    break
            map[entity] = id2idx
            # print(f'GET ENTITY GRAPH: entity2idx = {id2idx}')
            # 2. 结合当前entity，他的neighbors和id2idx, 构造adj
            adj = np.zeros((E,E))
            # seed = set(self.adj_entity[entity]+[entity])
            seed = set(id2idx.keys())
            for i in list(id2idx.keys()):
                # 给entity自己添加neighbors
                adj[0][id2idx[i]] = 1 # 0和i
                adj[id2idx[i]][0] = 1 # i和0
                adj[id2idx[i]][id2idx[i]] = 1 # i和i
                for j in list(set(self.adj_entity[i]) & seed):
                    adj[id2idx[i]][id2idx[j]] = 1
                    adj[id2idx[j]][id2idx[i]] = 1
            # print(f'GET ENTITY GRAPH: adj = {adj}')
            adjs[entity] = adj

        # 2. get the E neighbors of the node
        neighbors = {}
        # delete the duplicate in adj_entity and add 0 to E
        for entity in range(len(self.adj_entity)):
            tmp = list(map[entity].keys())
            for i in range(E-len(tmp)):
                tmp.extend([0]) # 0 代表不存在这个entity
            neighbors[entity] = tmp
        assert len(neighbors.keys()) == len(self.adj_entity)

        cached_graph = {}
        for entity in range(len(self.adj_entity)):
            cached_graph[entity] = ( neighbors[entity], adjs[entity])
        # print('neighbors[0]',neighbors[0])
        # print('neighbors[0]',adjs[0][1])
        # print('neighbors[1]',neighbors[1])
        # print('neighbors[1]',adjs[1][1])
        # print('neighbors[2]',neighbors[2])
        # print('neighbors[2]',adjs[2][1])
        return cached_graph
    
    def get_cached_graph(self, node):
        # import time 
        # cached_time = time.time()
        if self.cached_graph is None:
            with open(self.cached_graph_file,'rb') as fin:
                self.cached_graph = pickle.load(fin)
            # print(f'cached_graph shape= {self.cached_graph.shape}')
            # print(f'keys = {self.cached_graph.keys()}') #
        node, adj = self.cached_graph[node] # 这里node的节点序号非常大，肯定超过了movielens 数据集的id数目
        # print(f'cached_time = {time.time()-cached_time}')
     
        # num_neighbors = np.sum(node.numpy() != 0)

        # print(f'node.shape {np.array(node).shape}')
        # print(f'adj.shape {adj.shape}')
        node = torch.Tensor(np.array(node)).to(self.args.device)
        adj = torch.Tensor(adj).to(self.args.device)
        
        return node, adj
    
    def get_seq_graph(self, seq):
        """
        :param seq: a list of nodes [l] # 单个profile（不是整个batch）的node节点
        :return: seq_neighbor [L x E] seq_adjs [L x E x E] # E是节点的邻居数目
        """
        assert len(seq) <= self.max_seq_length

        neighbors, adjs = [], []
        # import time
        # seq_time  = time.time()
        # print('get_seq_graph', time.time())
        for s in seq: # 获得每一个profile每一个item的neighbors和adj
            '''
            get_cached_graph
            '''
            #n, adj, _ = self.get_graph(s)
            if s == -1:
                break
            n, adj = self.get_cached_graph(s)
            neighbors.append(n.unsqueeze(0))
            adjs.append(adj.unsqueeze(0))
        #print('finish get_seq_graph', time.time()-seq_time)
        E, L, l = self.max_nodes, self.max_seq_length, len(adjs)
        seq_adjs = torch.zeros((L, E, E)).to(self.args.device)
        seq_neighbors = torch.zeros((L, E)).long().to(self.args.device)

        seq_adjs[:l] = torch.cat(adjs, dim=0)  # [l x E x E]
        seq_neighbors[:l] = torch.cat(neighbors, dim=0)  # [l x E]
        #print('finish get_seq_graph2', time.time()-seq_time)
        return seq_neighbors, seq_adjs

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input [N x L x E x d]
        # [N x L x E x E]
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphEncoder(Module):
    def __init__(self, adj_entity, args):
        super(GraphEncoder, self).__init__()
        self.args = args
        # KG embedding
        print('==========KGE embedding loading==============')
        print('embedding is generated from KGE.py [transE]')
        KG_VEC = {'ml-1m':'ml1m-kg1m','ml-20m':'ml20m-kg500k','bx':'bx-kg150k','lastfm':'lastfm-kg15k'}
        embeddings = torch.load(f'data/kg/{KG_VEC[args.dataset]}/ent_embedding.pt')
        n_entity = embeddings.shape[0] # {movie-lens 1M: 182011}
        entity_emb = embeddings.shape[1] # {movie-lens 1M: 50}
        print(f'entity embedding size [{entity_emb}], num of entities [{n_entity}]')

        self.cached_graph_file = f'{args.processed_path}_{args.dataset}'
        self.adj_entity = adj_entity
        self.max_node = args.max_aggre_neighbor_entity
        self.max_seq_length = args.episode_length
        self.entity_emb = entity_emb
        self.embedding = nn.Embedding(n_entity,entity_emb)
        
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding = self.embedding.from_pretrained(embeddings,freeze=args.fix_emb)
        

        # 1. graphg构造器 GraphConstructer
        self.constructor = GraphConstructer(adj_entity = self.adj_entity, max_nodes=self.max_node, max_seq_length=self.max_seq_length, cached_graph_file = self.cached_graph_file, args = self.args)
        # 2. graph卷积层 GraphConvolution
        from torch_geometric.nn import TransformerConv, GCNConv,GATConv
        indim, outdim = entity_emb, args.gcn_hidden
        self.gnns = nn.ModuleList()
        for l in range(args.gcn_layer):
            # self.gnns.append(GraphConvolution(indim, outdim)) # gcn
            self.gnns.append(TransformerConv(indim, outdim))
            # self.gnns.append(GATConv(indim, outdim))
            # self.gnns.append(GCNConv(indim, outdim))
            # self.gnns.append(torch.nn.Linear(indim, outdim))

            indim = outdim
    
    def forward(self, seq):
        """
        :param seq: [N x l] ;candi:[N x K] # N 代表Batch, l代表当前的time_step长度
        :return: [N x L x d]
        """

        batch_seq_adjs = []
        batch_seq_neighbors = []
        import time
        # print('seq', seq)
        for s in seq: # s:[l]
            # print('s' , s)
            neighbors, adj = self.constructor.get_seq_graph(s) # neighbors是 [L x E] 而不是 [l x E]，L>l的部分用全0填充
            batch_seq_neighbors.append(neighbors[None, :])
            batch_seq_adjs.append(adj[None, :])
            # print('neighbors', neighbors.shape)
            # print('adj', adj.shape)
        
        input_neighbors_ids = torch.cat(batch_seq_neighbors, dim=0) # [N x L x E]
        input_adjs = torch.cat(batch_seq_adjs, dim=0)   # [N x L x E x E]
        
        input_state = self.embedding(input_neighbors_ids)  # [N x L x E x d]
        # print(f'input_state.shape = {input_state.shape}')
        # print(f'input_adjs.shape = {input_adjs.shape}')
        # here! no GNN
        from torch_geometric.data import Data, DataLoader
        from torch_geometric.utils.sparse import dense_to_sparse
        #print('Start Conv')
        input_adjs = input_adjs.view(-1,self.max_node,self.max_node)
        input_state = input_state.view(-1,self.max_node,self.entity_emb)
        data_list = []
        
        # _input_adj, _  = dense_to_sparse(inpu_adjs)
        for i in range(input_adjs.shape[0]):
            # to sparse tensor
            _input_adj, _ = dense_to_sparse(input_adjs[i])
            data_list.append(Data(x = input_state[i], edge_index = _input_adj ))
        loader = DataLoader(data_list, batch_size=input_adjs.shape[0])
        
        # for batch in loader:
        #     print(f'batch.x = {batch.x.shape}')
        #     print(f'batch.edge_index = {batch.edge_index.shape}')
        
        for batch in loader:
            for gnn in self.gnns:
                output_state = gnn(batch.x, batch.edge_index)
                output_state = F.relu(output_state)
                input_state = output_state
                
        # for gnn in self.gnns:
        #     output_state = gnn(input_state, input_adjs)
        #     input_state = output_state
        
        output_state = output_state.view(-1, self.max_seq_length, self.max_node, self.args.gcn_hidden)
        seq_embeddings = output_state[:, :, :1, :].contiguous().squeeze(2)  # [N x L x 1 x d] --> squeeze --> [N x L x d]
        # here! no GNN! 
        # seq_embeddings = input_state[:, :, :1, :].contiguous().squeeze(2)
        # print(f'seq_embeddings = {seq_embeddings.shape}')
        return seq_embeddings

if __name__=='__main__':
    
    from data import preprocess
    n_user, n_item, rating, unpopular_items, popular_items\
            ,  n_entity, n_relation, kg, adj_entity, adj_relation = preprocess.load_data(dataset = 'movie', kg_neighbor_size = 32)
    n_entity = len(adj_entity.keys())
    gc = GraphConstructer(adj_entity, max_node = 40, max_seq_length = 32)
    