## Random

# # target random (r)
# python Env_sage.py --gpu 2 --random_method r --action_size 100 --seed 123


# # target neighbor (n)
# python Env_sage.py --gpu 2 --random_method n --action_size 100 --seed 123
# python Env_sage.py --gpu 2 --random_method n --action_size 200 --seed 123
# python Env_sage.py --gpu 2 --random_method n --action_size 300 --seed 123
# python Env_sage.py --gpu 2 --random_method n --action_size 400 --seed 123
# python Env_sage.py --gpu 2 --random_method n --action_size 500 --seed 123
# python Env_sage.py --gpu 2 --random_method n --action_size 600 --seed 123
# python Env_sage.py --gpu 2 --random_method n --action_size 700 --seed 123
# # random neighbor (rn)
# python Env_sage.py --gpu 2 --random_method rn --action_size 100 --seed 123
# python Env_sage.py --gpu 2 --random_method rn --action_size 200 --seed 123
# python Env_sage.py --gpu 2 --random_method rn --action_size 300 --seed 123
# python Env_sage.py --gpu 2 --random_method rn --action_size 400 --seed 123
# python Env_sage.py --gpu 2 --random_method rn --action_size 500 --seed 123
# python Env_sage.py --gpu 2 --random_method rn --action_size 600 --seed 123
# python Env_sage.py --gpu 2 --random_method rn --action_size 700 --seed 123


## RL

# agent select
# python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 100 --seed 123
# python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 200 --seed 123
# python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 300 --seed 123
# python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 400 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 500 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 600 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 700 --seed 123

python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.1 --action_size 700 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.2 --action_size 700 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.3 --action_size 700 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.4 --action_size 700 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.5 --action_size 700 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.6 --action_size 700 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.7 --action_size 700 --seed 123
python framework.py --gpu 2 --candidate_mode agent_select --bandit_ratio 0.8 --action_size 700 --seed 123
# nearest neighbor
# python framework.py --gpu 2 --candidate_mode nearest_neighbor --action_size 100 --seed 123
# python framework.py --gpu 2 --candidate_mode nearest_neighbor --action_size 200 --seed 123
# python framework.py --gpu 2 --candidate_mode nearest_neighbor --action_size 300 --seed 123
# python framework.py --gpu 2 --candidate_mode nearest_neighbor --action_size 400 --seed 123
python framework.py --gpu 2 --candidate_mode nearest_neighbor --action_size 500 --seed 123
python framework.py --gpu 2 --candidate_mode nearest_neighbor --action_size 600 --seed 123
python framework.py --gpu 2 --candidate_mode nearest_neighbor --action_size 700 --seed 123
# target neighbor
# python framework.py --gpu 2 --candidate_mode target_neighbor --action_size 100 --seed 123
# python framework.py --gpu 2 --candidate_mode target_neighbor --action_size 200 --seed 123
# python framework.py --gpu 2 --candidate_mode target_neighbor --action_size 300 --seed 123
# python framework.py --gpu 2 --candidate_mode target_neighbor --action_size 400 --seed 123
python framework.py --gpu 2 --candidate_mode target_neighbor --action_size 500 --seed 123
python framework.py --gpu 2 --candidate_mode target_neighbor --action_size 600 --seed 123
python framework.py --gpu 2 --candidate_mode target_neighbor --action_size 700 --seed 123
# kg_item
# python framework.py --gpu 2 --candidate_mode kg_item --action_size 100 --seed 123
# python framework.py --gpu 2 --candidate_mode kg_item --action_size 200 --seed 123
# python framework.py --gpu 2 --candidate_mode kg_item --action_size 300 --seed 123
# python framework.py --gpu 2 --candidate_mode kg_item --action_size 400 --seed 123
python framework.py --gpu 2 --candidate_mode kg_item --action_size 500 --seed 123
python framework.py --gpu 2 --candidate_mode kg_item --action_size 600 --seed 123
python framework.py --gpu 2 --candidate_mode kg_item --action_size 700 --seed 123
# all_item
# python framework.py --gpu 2 --candidate_mode all_item --action_size 100 --seed 123
# python framework.py --gpu 2 --candidate_mode all_item --action_size 200 --seed 123
# python framework.py --gpu 2 --candidate_mode all_item --action_size 300 --seed 123
# python framework.py --gpu 2 --candidate_mode all_item --action_size 400 --seed 123
python framework.py --gpu 2 --candidate_mode all_item --action_size 500 --seed 123
python framework.py --gpu 2 --candidate_mode all_item --action_size 600 --seed 123
python framework.py --gpu 2 --candidate_mode all_item --action_size 700 --seed 123
