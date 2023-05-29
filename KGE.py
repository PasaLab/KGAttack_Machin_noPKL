import pandas as pd
import numpy
import os
import copy
import time

import torch
from torch.optim import Adam
from torchkge.data_structures import KnowledgeGraph
from torchkge.evaluation import LinkPredictionEvaluator,TripletClassificationEvaluator
from torchkge.models import TransEModel
from torchkge.utils import Trainer, MarginLoss

FILE = {'ml-1m':'ml1m-kg1m','ml-20m':'ml20m-kg500k','Book-Crossing':'bx-kg150k'}
class KGE(object):
    def __init__(self, dataset_name = 'ml-20m'):
        self.dataset_name = dataset_name
        self._id_map_path = f'data/kg/{FILE[dataset_name]}/item_id2entity_id.txt'
        self.emb_save_path = f'data/kg/{FILE[dataset_name]}/ent_embedding.pt'
        self._kg_path = f'data/kg/{FILE[dataset_name]}/kg.txt'
        self.item_index_old2new = {}
        self.entity_id2index  = {}
        self.relation_id2index = {}

        self.read_item_index_to_entity_id_file()
        kg_df = self.load_kg()
        self.train_KGE(kg_df)
    def read_item_index_to_entity_id_file(self):
        # file = './data/' + dataset + '/item_index2entity_id.txt'
        
        print('reading item index to entity id file: ' + self._id_map_path + ' ...')
        i = 0
        for line in open(self._id_map_path, encoding='utf-8').readlines():
            item_index = str(line.strip().split('\t')[0]) if self.dataset_name == 'Book-Crossing' else int(line.strip().split('\t')[0])
            satori_id = int(line.strip().split('\t')[1])
            self.item_index_old2new[item_index] = i
            self.entity_id2index[satori_id] = i
            i += 1
    
    def load_kg(self):
        kg = []
        num_indexed_entity = len(self.entity_id2index)
        num_indexed_relation = 0
        with open(self._kg_path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                array = line.strip().split('\t')
                head_old = int(array[0])
                relation_old = array[1]
                tail_old = int(array[2])
                
                if head_old not in self.entity_id2index:
                    self.entity_id2index[head_old] = num_indexed_entity
                    num_indexed_entity += 1
                head = self.entity_id2index[head_old]

                if tail_old not in self.entity_id2index:
                    self.entity_id2index[tail_old] = num_indexed_entity
                    num_indexed_entity += 1
                tail = self.entity_id2index[tail_old]

                if relation_old not in self.relation_id2index:
                    self.relation_id2index[relation_old] = num_indexed_relation
                    num_indexed_relation += 1
                relation = self.relation_id2index[relation_old]
                kg.append((head, tail, relation))
        print('number of entities (containing items): %d' % num_indexed_entity)
        print('number of relations: %d' % num_indexed_relation)
        kg_df = pd.DataFrame(kg, columns=['from', 'to', 'rel'])
        
        return kg_df
    
    def train_KGE(self, kg_df):
        # Load dataset
        kg_object = KnowledgeGraph(kg_df)
        kg_train, kg_val, kg_test = kg_object.split_kg(validation=True)
        # Define some hyper-parameters for training
        emb_dim = 50
        lr = 0.0004
        margin = 0.5
        n_epochs = 1000
        batch_size = 32768
        # Define the model and criterion
        model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel,
                            dissimilarity_type='L2')
        criterion = MarginLoss(margin)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        trainer = Trainer(model, criterion, kg_train, n_epochs, batch_size,
                        optimizer=optimizer, sampling_type='bern', use_cuda='all',)

        trainer.run()

        print('Link prediction Evaluate')
        evaluator = LinkPredictionEvaluator(model, kg_test)
        evaluator.evaluate(200)
        evaluator.print_results()
        
        print('Get entity embedding')
        ent_emb, rel_emb = model.get_embeddings()
        torch.save(ent_emb, self.emb_save_path)
        _ent_emb = torch.load(self.emb_save_path)
        print(f'Load ent emb shape = {_ent_emb.shape}')


if __name__ == '__main__':
    kge = KGE(dataset_name = 'Book-Crossing') # ml-1m, Book-Crossing
    

            

