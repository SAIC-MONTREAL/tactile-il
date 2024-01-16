#!/bin/bash
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=80 random_seed=144
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=80 random_seed=145
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=80 random_seed=146
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=96 random_seed=144
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=96 random_seed=145
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=96 random_seed=146
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=112 random_seed=144
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=112 random_seed=145
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=112 random_seed=146
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=128 random_seed=144
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=128 random_seed=145
python3 ${PWD}/train_vanilla_bc.py dataset=single-plate-kin action_distribution="mixture" dataset.dataset_config.n_max_episodes=128 random_seed=146