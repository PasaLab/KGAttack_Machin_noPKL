import sys
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy
from kgcn.aggregator import Aggregator

class KGCN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, num_item, item_num_in_kg, kg, args, adj_entity, adj_relation, device):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.num_item = num_item
        self.item_num_in_kg = item_num_in_kg
        # self.n_iter = args.n_iter
        self.device = device
        self.n_iter = args.kgcn_n_iter
        self.batch_size = args.batch_size
        # self.dim = args.dim
        self.dim = args.kgcn_dim
        # self.n_neighbor = args.neighbor_sample_size
        self.n_neighbor = 8
        self.kg = kg
        # self.device = device
        self.args = args
        self.aggregator = Aggregator(self.batch_size, self.dim, args.kgcn_aggregator)
        self._gen_adj()

        self.item_embedding = np.random.rand(num_item, self.dim)
        self.ent_embedding = np.random.rand(num_ent+1, self.dim)
        
        # set the item and ent embedding the same if index < self.item_num_in_kg
        self.ent_embedding[:item_num_in_kg] = self.item_embedding[:item_num_in_kg]
        self.item = torch.nn.Embedding.from_pretrained(torch.Tensor(self.item_embedding),freeze = False)
        self.ent = torch.nn.Embedding.from_pretrained(torch.Tensor(self.ent_embedding), freeze = False)
        self.usr = torch.nn.Embedding(num_user, self.dim)
        # self.item = torch.nn.Embedding(num_item, args.dim)
        # self.ent = torch.nn.Embedding(num_ent+1, args.dim)
        self.rel = torch.nn.Embedding(num_rel+1, self.dim)
        

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent+1, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent+1, self.n_neighbor, dtype=torch.long)
        
        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
                
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])
        
        # add a dummy entity whose neighbor entities and relations are dummy entity and dummy relations.
        self.adj_ent[self.num_ent] = torch.LongTensor([self.num_ent]*self.n_neighbor)
        self.adj_rel[self.num_ent] = torch.LongTensor([self.num_rel]*self.n_neighbor)
        
    def forward(self, u, v):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.view((-1, 1))
        v = v.view((-1, 1))
        
        # [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim = 1)
        
        entities, relations = self._get_neighbors(v)
        
        item_embeddings = self._aggregate(user_embeddings, entities, relations)
        
        scores = (user_embeddings * item_embeddings).sum(dim = 1)
            
        return torch.sigmoid(scores)
    
    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''

        entities = [v.clone()]

        # we change the seed, if v (i.e., items) is not in KG, then we give it a dummy entity ID = self.num_ent
        for i in range(len(entities[0])):
            if entities[0][i] >= self.item_num_in_kg:
                entities[0][i] = self.num_ent
        
        relations = []
        
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        
        # change the seed back to the v
        entities[0][:] = v
        return entities, relations
    
    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.item(entities[0]) if i==0 else self.ent(entities[i]) for i in range(len(entities))]
        # the seed is in item, so we should use item embedding
        relation_vectors = [self.rel(relation) for relation in relations]
        
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid
            
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        
        return entity_vectors[0].view((self.batch_size, self.dim))