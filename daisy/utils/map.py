import os
import numpy as np
import copy
from typing import Dict, List, Tuple, Callable, Any
import pickle
from collections import defaultdict
from tqdm import tqdm

KGFILE = {'ml-1m':'ml1m-kg1m','ml-20m':'ml20m-kg500k','bx':'bx-kg150k','lastfm':'lastfm-kg15k'}
# entity_id2index = dict()
# relation_id2index = dict()
# item_index_old2new = dict()
def neaten_id(dataset, rating_df, kg_neighbor_size):
    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()
    # read_item_index_to_entity_id_file
    file = './data/kg/' + KGFILE[dataset] + '/item_id2entity_id.txt'
    item_set = set(rating_df['item'].unique())
    del_item = []
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = str(line.strip().split('\t')[0]) if dataset == 'bx' else int(line.strip().split('\t')[0])
        satori_id = int(line.strip().split('\t')[1])

        if item_index in item_set:
            item_index_old2new[item_index] = i
            entity_id2index[satori_id] = i
            i += 1
        else:
            del_item.append(item_index)
    assert len(item_index_old2new) == len(entity_id2index)
    print(f'Num of deleted items in map file which not appears in datasets [{len(del_item)}]')
    print(f'Final item2KG map dict [{len(item_index_old2new)}]')

    # clean id according to map dict
    original_items = rating_df['item'].unique()
    original_users = rating_df['user'].unique()
    num_items_in_kg = len(item_index_old2new)
    num_indexed_items = len(item_index_old2new)

    for _, item in enumerate(original_items):
        if item not in item_index_old2new:
            item_index_old2new[item] = num_indexed_items
            num_indexed_items += 1
    user_map = {user: idx for idx, user in enumerate(original_users)}
    item_map = {item: item_index_old2new[item] for idx, item in enumerate(original_items)}
    print("Reindex dataframe...")
    rating_df['item'] = rating_df['item'].apply(lambda item:item_map[item])
    rating_df['user'] = rating_df['user'].apply(lambda user:user_map[user])

    print('Load Knowledge Graph')
    kg = []
    num_indexed_entity = len(entity_id2index)
    num_indexed_relation = 0
    _kg_path = f'data/kg/{KGFILE[dataset]}/kg.txt'
    with open(_kg_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            array = line.strip().split('\t')
            head_old = int(array[0])
            relation_old = array[1]
            tail_old = int(array[2])
            
            if head_old not in entity_id2index:
                entity_id2index[head_old] = num_indexed_entity
                num_indexed_entity += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = num_indexed_entity
                num_indexed_entity += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = num_indexed_relation
                num_indexed_relation += 1
            relation = relation_id2index[relation_old]
            kg.append((head, relation, tail))
    print('number of entities (containing items): %d' % num_indexed_entity)
    print('number of relations: %d' % num_indexed_relation)
    print('number of triples:',len(kg))

    print('Undirected KG')
    kg_dict = defaultdict(list)
    for head_id, relation_id, tail_id in kg:
        kg_dict[head_id].append((relation_id, tail_id))
        kg_dict[tail_id].append((relation_id, head_id))
    
    print('Get KG adj list')
    adj_entity, adj_relation,degree_item = [None for _ in range(num_indexed_entity)], [None for _ in range(num_indexed_entity)],{}
    for entity_id in range(num_indexed_entity):
        neighbors = kg_dict[entity_id]
        n_neighbor = len(neighbors)
        if entity_id < num_items_in_kg:
            degree_item[entity_id] = n_neighbor 
        sample_indices = np.random.choice(range(n_neighbor), size=kg_neighbor_size, replace=n_neighbor < kg_neighbor_size)
        adj_relation[entity_id] = [neighbors[i][0] for i in sample_indices]
        adj_entity[entity_id] = [neighbors[i][1] for i in sample_indices]
    
    return rating_df, num_items_in_kg, num_indexed_entity, num_indexed_relation, kg, kg_dict, adj_entity, adj_relation, degree_item

def get_unpopular(df, num_items_in_kg, unpop_ratio, pop_ratio = 0.3):
    item_count = [0 for i in range(num_items_in_kg)]
    unpopular_items = []
    popular_items = []
    for row in df.itertuples():
        item_id = getattr(row, 'item')
        if item_id < num_items_in_kg:
            item_count[item_id] += 1
    print("Get the unpopular item (item in KG)")
    item_count_sort = sorted(item_count, reverse=True)
    popular_line = item_count_sort[int(len(np.nonzero(item_count_sort)[0])*pop_ratio)]
    unpopular_line = item_count_sort[int(len(np.nonzero(item_count_sort)[0])*unpop_ratio)]
    print(f'popular_line: {popular_line}, unpopular_line: {unpopular_line}')
    # print(f'Unpopular_line: {unpopular_line}')
    for i in range(num_items_in_kg):
        if item_count[i]<=unpopular_line:
            unpopular_items.append(i)
        if item_count[i]>=popular_line:
            popular_items.append(i)
    print(f'popular item num: {len(popular_items)}, unpopular item num: {len(unpopular_items)}')
    # print(f'Unpopular item num: {len(unpopular_items)}')
    return unpopular_items,popular_items

def find_neighbor_items(processed_path, adj_entity, item_count,hop):
    # item_count: item_num_in_kg
    if os.path.exists(processed_path):
        print('adj_item file exist')
        with open(processed_path,'rb') as fin:
            adj_item = pickle.load(fin)
            # adj_item = json.load(fin)
        #candi_dict = utils.pickle_load('./neighbors.pkl')
        #print(type(adj_item))
        a = 9999
        b = -1
        cnt = 0
        count = []
        for i in range(item_count):
            count.append(len(adj_item[i]))
            a = min(a,len(adj_item[i]))
            b = max(b,len(adj_item[i]))
            if len(adj_item[i]) == 0:
                cnt+=1
        print(f'kg_hop H = {hop}, have checked, adj_item min len = {a}, max len {b}, zero cnt = {cnt}, now return adj_item') # ml-20m: adj_item min len = 1, zero cnt = 0
        print(f'avg of {np.mean(count)}')
        # exit()
        import seaborn as sns
        import matplotlib.pyplot as plt
        plot = sns.kdeplot(count)
        plt.savefig('./ml-20m_kg_hop3.png')
        # from collections import Counter
        # c = Counter(count)
        # print(c)
        # print(c.most_common())
        return adj_item
    else:
        # 给定kg 三元组，给entity(item)找到他们的对应的neighbor entity(item):
        adj_item = {}
        item_set = set([i for i in range(item_count)])
        for item_new_id in range(item_count):
            adj_item[item_new_id] = set()
        for item_new_id in tqdm(range(item_count)):
            seed = adj_entity[item_new_id]
            for k in range(hop):
                tmp = copy.deepcopy(seed)
                for item in tmp:
                    seed.extend(adj_entity[item])
                    seed = list(set(seed))
                # seed = list(set(seed))
            seedset = set(seed)
            #print(f'{item_new_id}before interact len{len(seedset)}')
            seedset = seedset & item_set 
            #print(f'{item_new_id}after interact len{len(seedset)}')
            adj_item[item_new_id]=list(seedset)
        a = 9999
        cnt = 0
        for i in range(item_count):
            if a > len(adj_item[i]):
                a = len(adj_item[i])
            if len(adj_item[i]) == 0:
                cnt+=1
        print(f'have checked, adj_item min len = {a}, zero cnt = {cnt}, now save adj_item')
        with open(processed_path,'wb') as fin:
            pickle.dump(adj_item, fin)
        return adj_item