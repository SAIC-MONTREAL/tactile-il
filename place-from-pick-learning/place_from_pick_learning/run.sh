#!/bin/bash
python3 ${PWD}/train_vanilla_bc.py dataset=triple-plates-noisy action_distribution="mixture" random_seed=144
python3 ${PWD}/train_vanilla_bc.py dataset=triple-plates-noisy action_distribution="mixture" random_seed=145
python3 ${PWD}/train_vanilla_bc.py dataset=triple-plates-noisy action_distribution="mixture" random_seed=146