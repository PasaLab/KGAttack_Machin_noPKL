import os
import gc
import re
import json
import random
import numpy as np
import pandas as pd
import scipy.io as sio

from collections import defaultdict


def load_rate(src='ml-100k', prepro='origin', binary=True, pos_threshold=None, level='ui'):
    """
    Method of loading certain raw data
    Parameters
    ----------
    src : str, the name of dataset
    prepro : str, way to pre-process raw data input, expect 'origin', f'{N}core', f'{N}filter', N is integer value
    binary : boolean, whether to transform rating to binary label as CTR or not as Regression
    pos_threshold : float, if not None, treat rating larger than this threshold as positive sample
    level : str, which level to do with f'{N}core' or f'{N}filter' operation (it only works when prepro contains 'core' or 'filter')
    Returns
    -------
    df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
    user_num : int, the number of users
    item_num : int, the number of items
    """
    df = pd.DataFrame()
    # which dataset will use
    if src == 'ml-100k':
        df = pd.read_csv(f'./data/ds/{src}/u.data', sep='\t', header=None,
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')

    elif src == 'ml-1m':
        df = pd.read_csv(f'./data/ds/{src}/ratings.dat', sep='::', header=None, 
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')
        pos_threshold = 4
        # only consider rating >=4 for data density
        # df = df.query('rating >= 4').reset_index(drop=True).copy()

    elif src == 'ml-10m':
        df = pd.read_csv(f'./data/ds/{src}/ratings.dat', sep='::', header=None, 
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')
        pos_threshold = 4
        
        # df = df.query('rating >= 4').reset_index(drop=True).copy()

    elif src == 'ml-20m':
        df = pd.read_csv(f'./data/ds/{src}/ratings.csv')
        df.rename(columns={'userId':'user', 'movieId':'item'}, inplace=True)
        pos_threshold = 4
        # df = df.query('rating >= 4').reset_index(drop=True)

    elif src == 'netflix':
        cnt = 0
        tmp_file = open(f'./data/ds/{src}/training_data.csv', 'w')
        tmp_file.write('user,item,rating,timestamp' + '\n')
        for f in os.listdir(f'./data/ds/{src}/training_set/'):
            cnt += 1
            if cnt % 5000 == 0:
                print(f'Finish Process {cnt} file......')
            txt_file = open(f'./data/ds/{src}/training_set/{f}', 'r')
            contents = txt_file.readlines()
            item = contents[0].strip().split(':')[0]
            for val in contents[1:]:
                user, rating, timestamp = val.strip().split(',')
                tmp_file.write(','.join([user, item, rating, timestamp]) + '\n')
            txt_file.close()

        tmp_file.close()

        df = pd.read_csv(f'./data/ds/{src}/training_data.csv')
        df['rating'] = df.rating.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    elif src == 'lastfm':
        # user_artists.dat
        df = pd.read_csv(f'./data/ds/{src}/user_artists.dat', sep='\t')
        df.rename(columns={'userID': 'user', 'artistID': 'item', 'weight': 'rating'}, inplace=True)
        # treat weight as interaction, as 1
        df['rating'] = 1.0
        # fake timestamp column
        df['timestamp'] = 1

    elif src == 'bx':
        df = pd.read_csv(f'./data/ds/{src}/BX-Book-Ratings.csv',index_col=False, sep=';', names=['user', 'item', 'rating'], header=0, encoding='cp1252')
        # df = pd.read_csv(f'./data/ds/{src}/BX-Book-Ratings.csv', delimiter=";", encoding="cp1252")
        # df.rename(columns={'User-ID': 'user', 'ISBN': 'item', 'Book-Rating': 'rating'}, inplace=True)
        # fake timestamp column
        df['timestamp'] = 1
        pos_threshold = 0

    elif src == 'pinterest':
        # TODO this dataset has wrong source URL, we will figure out in future
        pass

    elif src == 'amazon-cloth':
        df = pd.read_csv(f'./data/ds/{src}/ratings_Clothing_Shoes_and_Jewelry.csv', 
                         names=['user', 'item', 'rating', 'timestamp'])

    elif src == 'amazon-electronic':
        df = pd.read_csv(f'./data/ds/{src}/ratings_Electronics.csv', 
                         names=['user', 'item', 'rating', 'timestamp'])

    elif src == 'amazon-book':
        df = pd.read_csv(f'./data/ds/{src}/ratings_Books.csv', 
                         names=['user', 'item', 'rating', 'timestamp'], low_memory=False)
        df = df[df['timestamp'].str.isnumeric()].copy()
        df['timestamp'] = df['timestamp'].astype(int)

    elif src == 'amazon-music':
        df = pd.read_csv(f'./data/ds/{src}/ratings_Digital_Music.csv', 
                         names=['user', 'item', 'rating', 'timestamp'])

    elif src == 'epinions':
        d = sio.loadmat(f'./data/ds/{src}/rating_with_timestamp.mat')
        prime = []
        for val in d['rating_with_timestamp']:
            user, item, rating, timestamp = val[0], val[1], val[3], val[5]
            prime.append([user, item, rating, timestamp])
        df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])
        del prime
        gc.collect()

    elif src == 'yelp':
        json_file_path = f'./data/ds/{src}/yelp_academic_dataset_review.json'
        prime = []
        for line in open(json_file_path, 'r', encoding='UTF-8'):
            val = json.loads(line)
            prime.append([val['user_id'], val['business_id'], val['stars'], val['date']])
        df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])
        df['timestamp'] = pd.to_datetime(df.timestamp)
        del prime
        gc.collect()

    elif src == 'citeulike':
        user = 0
        dt = []
        for line in open(f'./data/ds/{src}/users.dat', 'r'):
            val = line.split()
            for item in val:
                dt.append([user, item])
            user += 1
        df = pd.DataFrame(dt, columns=['user', 'item'])
        # fake timestamp column
        df['timestamp'] = 1

    else:
        raise ValueError('Invalid Dataset Error')
    
    # set rating >= threshold as positive samples
    if pos_threshold is not None:
        df = df.query(f'rating >= {pos_threshold}').reset_index(drop=True)

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rating'] = 1.0

    # which type of pre-dataset will use
    print('pre-dataset')
    if prepro == 'origin':
        pass

    elif prepro.endswith('filter'):
        pattern = re.compile(r'\d+')
        filter_num = int(pattern.findall(prepro)[0])
        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        print('tmp1',tmp1)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        print('before prepro filter df',df)
        if level == 'ui':  
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()        
        else:
            raise ValueError(f'Invalid level value: {level}')
        print('after prepro filter df',df)
        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid level value: {level}')

        gc.collect()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')

    # encoding user_id and item_id
    # df['user'] = pd.Categorical(df['user']).codes
    # df['item'] = pd.Categorical(df['item']).codes

    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    print(f'Finish loading [{src}]-[{prepro}] dataset: user_num [{user_num}] item_num [{item_num}]')

    return df, user_num, item_num


def get_ur(df):
    """
    Method of getting user-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe
    Returns
    -------
    ur : dict, dictionary stored user-items interactions
    """
    ur = defaultdict(set)
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))

    return ur


def get_ir(df):
    """
    Method of getting item-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe
    Returns
    -------
    ir : dict, dictionary stored item-users interactions
    """
    ir = defaultdict(set)
    for _, row in df.iterrows():
        ir[int(row['item'])].add(int(row['user']))

    return ir


def build_feat_idx_dict(df:pd.DataFrame, 
                        cat_cols:list=['user', 'item'], 
                        num_cols:list=[]):
    """
    Method of encoding features mapping for FM
    Parameters
    ----------
    df : pd.DataFrame feature dataframe
    cat_cols : List, list of categorical column names
    num_cols : List, list of numeric column names
    Returns
    -------
    feat_idx_dict : Dictionary, dict with index-feature column mapping information
    cnt : int, the number of features
    """
    feat_idx_dict = {}
    idx = 0
    for col in cat_cols:
        feat_idx_dict[col] = idx
        idx = idx + df[col].max() + 1
    for col in num_cols:
        feat_idx_dict[col] = idx
        idx += 1
    print('Finish build feature index dictionary......')

    cnt = 0
    for col in cat_cols:
        for _ in df[col].unique():
            cnt += 1
    for _ in num_cols:
        cnt += 1
    print(f'Number of features: {cnt}')

    return feat_idx_dict, cnt


def convert_npy_mat(user_num, item_num, df):
    """
    method of convert dataframe to numpy matrix
    Parameters
    ----------
    user_num : int, the number of users
    item_num : int, the number of items
    df :  pd.DataFrame, rating dataframe
    Returns
    -------
    mat : np.matrix, rating matrix
    """
    mat = np.zeros((user_num, item_num))
    for _, row in df.iterrows():
        u, i, r = row['user'], row['item'], row['rating']
        mat[int(u), int(i)] = float(r)
    return mat


def build_candidates_set(test_ur, train_ur, item_pool, args, target_item=None ):
    """
    method of building candidate items for ranking
    Parameters
    ----------
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_pool : the set of all items
    candidates_num : int, the number of candidates
    Returns
    -------
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    candidates_num = args.max_candi_num
    num_eval_spy = args.num_eval_spy
    num_train_spy = args.num_train_spy
    test_ucands = defaultdict(list)

    for k, v in test_ur.items():
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
        sub_item_pool = item_pool - v - train_ur[k] # remove GT & interacted
        sample_num = min(len(sub_item_pool), sample_num)
        if sample_num == 0:
            samples = random.sample(v, candidates_num)
            test_ucands[k] = list(set(samples))
        else:
            samples = random.sample(sub_item_pool, sample_num)
            test_ucands[k] = list(v | set(samples))
        
        if target_item is not None:
            test_ucands[k] = list(set(test_ucands[k]) - set(target_item))
    # print(f'test_ucands = {len(test_ucands.keys())}, num_train_spy = {num_train_spy}, num_eval_spy = {num_eval_spy}')
    test_ucands_total = test_ucands.copy()
    def random_a_dict_and_sample_it( a_dictionary , a_number ): 
        _ = {}
        for k1 in random.sample( list( a_dictionary.keys() ) , a_number ):
            _[ k1 ] = a_dictionary[ k1 ]
        return _
    train_spy = random_a_dict_and_sample_it(test_ucands, num_train_spy)
    for u in train_spy.keys():
        del test_ucands[u]
    eval_spy = random_a_dict_and_sample_it(test_ucands, num_eval_spy)
    
    print(f'test ucands [{len(test_ucands_total)}] train_spy [{len(train_spy)}], eval_spy [{len(eval_spy)}]')
    return test_ucands_total, train_spy, eval_spy

def load_attack_data_get_attack_instances(attack_data):
    print('convert attack profiles to attack df')
    ui, user, item, r = {},[],[],[]
    for p in attack_data:
        u = p[0]
        for i in p[1:]:
            user.append(u)
            item.append(i)
            r.append(4)
    ui['user'] = user
    ui['item'] = item
    ui['rating'] = r
    attack_df = pd.DataFrame(ui)
    
    rating_dict = {}
    print("Store data into dictionary...")
    for row in attack_df.itertuples():
        user_id = getattr(row, 'user')
        item_id = getattr(row, 'item')
        rating = getattr(row, 'rating')
        if user_id not in rating_dict:
            rating_dict[user_id] = []
        rating_dict[user_id].append((item_id, rating))

    print('get attack fintune data')
    train_positives = []
    train_negatives = []
    all_items = set(range(self.num_items))
    for user_id in rating_dict:
        rated_items = set([record[0] for record in rating_dict[user_id]])
        all_negatives = all_items.difference(rated_items)
        for record in rating_dict[user_id]:
            item_id = record[0]
            rating = record[1]
            train_positives.append((user_id, item_id, rating))
            sample_items = random.sample(all_negatives, self.num_train_negatives)
            train_negatives.append(sample_items)
        assert len(train_positives) == len(train_negatives)
    print('get train instance')
    user_input, item_input, labels = [], [], []
    for i in range(len(train_positives)):
        record = train_positives[i]
        user = record[0]
        item = record[1]
        user_input.append(user)
        item_input.append(item)
        labels.append(1)
        for item in train_negatives[i]:
            user_input.append(user)
            item_input.append(item)
            labels.append(0)
    user_input = np.array(user_input)
    item_input = np.array(item_input)
    labels = np.array(labels)
    X_train = [user_input, item_input]
    Y_train = labels
    return X_train, Y_train