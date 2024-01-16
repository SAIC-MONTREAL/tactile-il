import hydra

from place_from_pick_learning.train_vanilla_bc import train_vanilla_bc


hydra_main_args = dict(config_path='cfgs', config_name='cfg')
if hydra.__version__ != '1.0.6':
    hydra_main_args['version_base'] = None

@hydra.main(**hydra_main_args)
def train_bc(cfg):
    train_vanilla_bc(cfg)


if __name__ == "__main__":
    train_bc()