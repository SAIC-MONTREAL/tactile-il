python3 control_place_from_pick.py \
    --model_dir "/home/rbslab/place-from-pick-learning/results/models/2023.02.10/action_distribution=mixture,dataset=triple-plates-noisy,random_seed=144_17-15-25" \
    --save_dir "/home/rbslab/place-from-pick-learning/results/control_runs/demo/" \
    --random_seed 144 \
    --n_steps_horizon 36 \
    --n_episodes 3 \
    --env_config_path "/home/rbslab/place-from-pick-learning/place_from_pick_learning/cfgs/env/demo.yaml" \