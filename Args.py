import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='test recommender')
    # Attack settings
    parser.add_argument('--candidate_mode', 
                        type=str, 
                        default='agent_select', 
                        help='agent_select/target_neighbor/all_item/kg_item/nearest_neighbor')
    parser.add_argument('--seed', 
                        type=int, 
                        default=123, 
                        help='')
    parser.add_argument('--target_bandit', 
                        type=float, 
                        default=0.2, 
                        help='')
    parser.add_argument('--unpop_ratio', 
                        type=float, 
                        default=0.9, 
                        help='')
    parser.add_argument('--bandit_ratio', 
                        type=float, 
                        default=0.2, 
                        help='')
    parser.add_argument('--action_size', 
                        type=int, 
                        default=500, 
                        help='')
    parser.add_argument('--attack_topk', 
                        type=int, 
                        default=20, 
                        help='(for RL reward) top number of recommend list')
    parser.add_argument('--reward_topk', 
                        type=int, 
                        default=20, 
                        help='(for RL reward) top number of recommend list')
    parser.add_argument('--attack_epochs', 
                        type=int, 
                        default=5, 
                        help='Attack fintune epochs')
    parser.add_argument('--num_target_item', 
                        type=int, 
                        default=10, 
                        help='Number of target items')
    parser.add_argument('--num_train_attacker', 
                        type=int, 
                        default=3, 
                        help='Number of attackers in RL exploring (injection) phase')
    parser.add_argument('--num_max_attacker', 
                        type=int, 
                        default=1000, 
                        help='Pre-defined number for NeuMF or Pinsage')
    parser.add_argument('--episode_length', 
                        type=int, 
                        default=16, 
                        help='Length of fake user profile')
    parser.add_argument('--max_episodes', 
                        type=int, 
                        default=50, 
                        help='number of episodes')
    parser.add_argument('--num_eval_spy', 
                        type=int, 
                        default=50, 
                        help='Length of eval spy')
    parser.add_argument('--num_train_spy', 
                        type=int, 
                        default=500, 
                        help='Length of train spy')
    parser.add_argument('--processed_path', 
                        type=str, 
                        default='processed_data', 
                        help='select dataset')
    parser.add_argument('--kg_neighbor_size', 
                        type=int, 
                        default=32, 
                        help='Num of KG neighbor entities')  
    parser.add_argument('--kg_hop', 
                        type=int, 
                        default=2, 
                        help='KG hop for item adj')                 
    # common setting
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    # env settings
    parser.add_argument('--problem_type', 
                        type=str, 
                        default='point', 
                        help='pair-wise or point-wise')
    parser.add_argument('--algo_name', 
                        type=str, 
                        default='NeuMF', 
                        help='algorithm to choose')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-1m', 
                        help='select dataset')
    parser.add_argument('--prepro', 
                        type=str, 
                        default='10filter', # 10filter
                        help='dataset preprocess op.: origin/Ncore/Nfilter')
    parser.add_argument('--level', 
                        type=str, 
                        default='u',
                        help='op.: ui/u/i')
    parser.add_argument('--topk', 
                        type=int, 
                        default=20, 
                        help='top number of recommend list')
    parser.add_argument('--test_method', 
                        type=str, 
                        default='tfo', 
                        help='method for split test,options: ufo/loo/fo/tfo/tloo')
    parser.add_argument('--val_method', 
                        type=str, 
                        default='tfo', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--test_size', 
                        type=float, 
                        default=.2, 
                        help='split ratio for test set')
    parser.add_argument('--val_size', 
                        type=float, 
                        default=.1, help='split ratio for validation set')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    parser.add_argument('--max_candi_num', 
                        type=int, 
                        default=1000, 
                        help='No. of MAX candidates item for predict')
    parser.add_argument('--candi_num', 
                        type=int, 
                        default=100, 
                        help='No. of candidates item for predict')
    parser.add_argument('--sample_method', 
                        type=str, 
                        default='uniform', 
                        help='negative sampling method mixed with uniform, options: item-ascd, item-desc')
    parser.add_argument('--sample_ratio', 
                        type=float, 
                        default=0, 
                        help='mix sample method ratio, 0 for all uniform')
    parser.add_argument('--init_method', 
                        type=str, 
                        default='', 
                        help='weight initialization method')
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=2,  # 30 for sage?
                        help='negative sampling number')
    parser.add_argument('--loss_type', 
                        type=str, 
                        default='CL', 
                        help='loss function type')
    parser.add_argument('--init_dim', 
                        type=int, 
                        default=16, 
                        help='No. of candidates item for predict')
    parser.add_argument('--dim1', 
                        type=int, 
                        default=16, 
                        help='No. of candidates item for predict')
    parser.add_argument('--dim2', 
                        type=int, 
                        default=16, 
                        help='No. of candidates item for predict')
    parser.add_argument('--sage_epoch', 
                        type=int, 
                        default=600, 
                        help='No. of candidates item for predict')
    # algo settings
    parser.add_argument('--rl_reward_type', 
                        type=str, 
                        default='hr', 
                        help='reward type to train RL: hr/ndcg')
    parser.add_argument('--factors', 
                        type=int, 
                        default=32, 
                        help='latent factors numbers in the model')
    parser.add_argument('--reg_1', 
                        type=float, 
                        default=0., 
                        help='L1 regularization')
    parser.add_argument('--reg_2', 
                        type=float, 
                        default=0.001, 
                        help='L2 regularization')
    parser.add_argument('--kl_reg', 
                        type=float, 
                        default=0.5, 
                        help='VAE KL regularization')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.5, 
                        help='dropout rate')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001, 
                        help='learning rate')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help='training epochs')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128, 
                        help='batch size for training')
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=2, 
                        help='number of layers in MLP model')
    parser.add_argument('--act_func', 
                        type=str, 
                        default='relu', 
                        help='activation method in interio layers')
    parser.add_argument('--out_func', 
                        type=str, 
                        default='sigmoid', 
                        help='activation method in output layers')
    parser.add_argument('--no_batch_norm', 
                        action='store_false', 
                        default=True, 
                        help='whether do batch normalization in interior layers')
    # kgcn setting
    parser.add_argument('--kgcn_aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--kgcn_epochs', type=int, default=10, help='the number of epochs')
    parser.add_argument('--kgcn_dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--kgcn_n_iter', type=int, default=1, help='number of iterations when computing entity representation')
    parser.add_argument('--kgcn_l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--kgcn_lr', type=float, default=5e-4, help='learning rate')
    # gnn setting
    parser.add_argument('--max_aggre_neighbor_entity', 
                        type=int, 
                        default=40, 
                        help='number of neighbors for one entity to aggregate in GNN')
    parser.add_argument('--gcn_hidden', 
                        type=int, 
                        default=50, 
                        help='hidden dimension in GNN')
    parser.add_argument('--gcn_layer', 
                        type=int, 
                        default=1, 
                        help='layer num in GNN')
    parser.add_argument('--fix_emb', 
                        action='store_false', 
                        default=True, 
                        help='whtat?')
    # actor setting
    parser.add_argument('--gru_hidden', 
                        type=int, 
                        default=50, 
                        help='embedding size of gru hidden layer')
    parser.add_argument('--actor_hidden', 
                        type=int, 
                        default=50, 
                        help='hidden dimension for actor network')
    parser.add_argument('--critic_hidden', 
                        type=int, 
                        default=50, 
                        help='hidden dimension for actor network')
    parser.add_argument('--gru_layer', 
                        type=int, 
                        default=1, 
                        help='layer size of gru to reprenent state')
    parser.add_argument('--actor_update', 
                        type=int, 
                        default=5, 
                        help='actor update time')
    # actor setting
    parser.add_argument('--critic_update', 
                        type=int, 
                        default=10, 
                        help='critic update time')
    # RL setting
    parser.add_argument('--discount', 
                        type=int, 
                        default=0.99, 
                        help='discount for reward')
    # Random setting
    parser.add_argument('--random_method', 
                        type=str, 
                        default='r', 
                        help='rn/n/r')
    args = parser.parse_args()

    return args