## Random

# # target random (r)
# python Env_sage.py --gpu 1 --random_method r --action_size 100 --seed 25

# # target neighbor (n)
# python Env_sage.py --gpu 1 --random_method n --action_size 100 --seed 25
# python Env_sage.py --gpu 1 --random_method n --action_size 200 --seed 25
# python Env_sage.py --gpu 1 --random_method n --action_size 300 --seed 25
# python Env_sage.py --gpu 1 --random_method n --action_size 400 --seed 25
# python Env_sage.py --gpu 1 --random_method n --action_size 500 --seed 25
# python Env_sage.py --gpu 1 --random_method n --action_size 600 --seed 25
# python Env_sage.py --gpu 1 --random_method n --action_size 700 --seed 25
# # random neighbor (rn)
# python Env_sage.py --gpu 1 --random_method rn --action_size 100 --seed 25
# python Env_sage.py --gpu 1 --random_method rn --action_size 200 --seed 25
# python Env_sage.py --gpu 1 --random_method rn --action_size 300 --seed 25
# python Env_sage.py --gpu 1 --random_method rn --action_size 400 --seed 25
# python Env_sage.py --gpu 1 --random_method rn --action_size 500 --seed 25
# python Env_sage.py --gpu 1 --random_method rn --action_size 600 --seed 25
# python Env_sage.py --gpu 1 --random_method rn --action_size 700 --seed 25

## RL

# agent select
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 100 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 200 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 300 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 400 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 600 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 700 --seed 25

python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.1 --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.3 --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.4 --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.5 --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.6 --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.7 --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode agent_select --bandit_ratio 0.8 --action_size 500 --seed 25
# nearest neighbor
python framework.py --gpu 1 --candidate_mode nearest_neighbor --action_size 100 --seed 25
python framework.py --gpu 1 --candidate_mode nearest_neighbor --action_size 200 --seed 25
python framework.py --gpu 1 --candidate_mode nearest_neighbor --action_size 300 --seed 25
python framework.py --gpu 1 --candidate_mode nearest_neighbor --action_size 400 --seed 25
python framework.py --gpu 1 --candidate_mode nearest_neighbor --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode nearest_neighbor --action_size 600 --seed 25
python framework.py --gpu 1 --candidate_mode nearest_neighbor --action_size 700 --seed 25
# target neighbor
python framework.py --gpu 1 --candidate_mode target_neighbor --action_size 100 --seed 25
python framework.py --gpu 1 --candidate_mode target_neighbor --action_size 200 --seed 25
python framework.py --gpu 1 --candidate_mode target_neighbor --action_size 300 --seed 25
python framework.py --gpu 1 --candidate_mode target_neighbor --action_size 400 --seed 25
python framework.py --gpu 1 --candidate_mode target_neighbor --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode target_neighbor --action_size 600 --seed 25
python framework.py --gpu 1 --candidate_mode target_neighbor --action_size 700 --seed 25
# kg_item
python framework.py --gpu 1 --candidate_mode kg_item --action_size 100 --seed 25
python framework.py --gpu 1 --candidate_mode kg_item --action_size 200 --seed 25
python framework.py --gpu 1 --candidate_mode kg_item --action_size 300 --seed 25
python framework.py --gpu 1 --candidate_mode kg_item --action_size 400 --seed 25
python framework.py --gpu 1 --candidate_mode kg_item --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode kg_item --action_size 600 --seed 25
python framework.py --gpu 1 --candidate_mode kg_item --action_size 700 --seed 25
# all_item
python framework.py --gpu 1 --candidate_mode all_item --action_size 100 --seed 25
python framework.py --gpu 1 --candidate_mode all_item --action_size 200 --seed 25
python framework.py --gpu 1 --candidate_mode all_item --action_size 300 --seed 25
python framework.py --gpu 1 --candidate_mode all_item --action_size 400 --seed 25
python framework.py --gpu 1 --candidate_mode all_item --action_size 500 --seed 25
python framework.py --gpu 1 --candidate_mode all_item --action_size 600 --seed 25
python framework.py --gpu 1 --candidate_mode all_item --action_size 700 --seed 25
