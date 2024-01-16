import argparse 
import os 
import re
import pickle 

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

def open_pkl(save_path):
    with open(save_path, "rb") as handle:
        stored_grasps = pickle.load(handle)
    return stored_grasps

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dirs', nargs='+', required=True, help='List of directory names, one directory name per model')
    args = parser.parse_args()
    return args

def get_avg_success_rate(model_dir):
    """Get average success rate per N episodes per M seeds."""        
    seed_dirs = [os.path.join(model_dir, name) for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir,name))]
    success_rates = []
    total_successes = []
    for seed_dir in seed_dirs:
        successes = []
        seed_dir = [os.path.join(seed_dir, v) for v in os.listdir(seed_dir) if v.endswith('.pkl')]
        for ep in seed_dir:
            data = open_pkl(ep)
            successes.append(data["success"])
            total_successes.append(data["success"])
        sr = sum(successes) / len(successes)
        success_rates.append(sr)

    mu_sr = (sum(success_rates) / len(success_rates))
    var_sr = sum([((x - mu_sr) ** 2) for x in success_rates]) / len(success_rates)
    std_sr = var_sr ** 0.5 + 1e-6

    frac = f"{sum(total_successes)} / {len(total_successes)}"
    return mu_sr, std_sr, frac
    
if __name__=="__main__":
    args = parse_args()

    # Process data per model
    models = {}
    for result_dir in args.result_dirs: 
        models[result_dir] = {"data": [], "frac": []}
        for name in os.listdir(result_dir):
            n_train_ep = int(re.search('(.+?)-train-episodes', name).group(1))
            mu_sr, std_sr, frac = get_avg_success_rate(os.path.join(result_dir, name))
            models[result_dir]["data"].append((n_train_ep, mu_sr, std_sr))
            models[result_dir]["frac"].append(frac)

        models[result_dir]["data"] = np.array(models[result_dir]["data"])

    # Plot success rate curve per model
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel('Amount of Demonstrations', fontsize=20)
    ax.set_ylabel('Success Rate [%]', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim([0, 1.0])

    for ii, (k,v) in enumerate(models.items()):
        name = k.split("/")[-1]
        data = v["data"]
        frac = v["frac"]      
        # Print results
        print(f"Model name: {name}")
        for jj in range(data.shape[0]):
            print(f"Training demonstrations: {data[jj, 0]}, Success: {frac[jj]} Mean success rate {data[jj, 1]}, Std success rate {data[jj, 2]}")

        # Sort
        data = data[data[:, 0].argsort()]
        x = data[:, 0]
        y = data[:, 1]
        std_y = data[:, 2]

        ax.plot(x, y, lw=3, label=name)
        ax.fill_between(x, y - std_y, y + std_y, facecolor=f'C{ii}', alpha=0.4)
        ax.legend(loc='lower right', fontsize=16)
    
    # Save figure
    plt.savefig('success_rate_curve.png', bbox_inches="tight")
    plt.savefig('success_rate_curve.pdf', bbox_inches="tight")
