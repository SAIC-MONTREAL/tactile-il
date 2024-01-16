python3 control_place_from_pick.py \
    --model_dir "/home/rbslab/place-from-pick-learning/results/models/2023.01.27/action_distribution=mixture,dataset.dataset_config.n_max_episodes=128,dataset=single-plate-kin,random_seed=146_22-14-54" \
    --save_dir "/home/rbslab/place-from-pick-learning/results/control_runs/kin-dataset_mixture-low-noise/128-train-episodes/146" \
    --random_seed 1466 \
    --n_steps_horizon 48 \
    --n_episodes 1 \
    --env_config_path "/home/rbslab/place-from-pick-learning/place_from_pick_learning/cfgs/env/no_sts.yaml" \