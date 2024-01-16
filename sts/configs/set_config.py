from omegaconf import DictConfig, OmegaConf
import hydra
import json
import sys
import os

@hydra.main(version_base=None, config_name="hydra", config_path=f"{os.environ['STS_PARENT_DIR']}/sts-cam-ros2/configs")
def set_cfg(cfg: DictConfig) -> None:
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    sts_parent_dir = os.environ["STS_PARENT_DIR"]


    # == SET WORKING CONFIG DIRECTORY ==

    # We must pop 'cfg_name' because it is not in the same directory 
    # as the other config files.
    if 'config_name' in cfg_dict.keys():
        cfg_name = cfg_dict.pop('config_name')
    # If 'config_name' is not one of the input arguments, use 'demo' as default.
    else:
        cfg_name = 'demo'
    config_dir = sts_parent_dir + f"/sts-cam-ros2/configs/{cfg_name}"
    
    working_cfg_dct = {}
    # Set working config directory
    with open(sts_parent_dir + f"/sts-cam-ros2/configs/working_config_dir.json", "r") as fd:
        working_cfg_dct = json.load(fd)

    working_cfg_dct["config_dir"] = config_dir

    # Write to file
    with open(sts_parent_dir + f"/sts-cam-ros2/configs/working_config_dir.json", "w") as fd:
        json.dump(working_cfg_dct, fd, indent=4)
        print(f"In working_config_dir.json, setting 'config_dir'={config_dir}")


    # == APPLY CONFIG CHANGES FOR THE REST OF THE INPUT ARGS ==

    loaded = {}
    # Load all json files before making modifications
    for filename in cfg_dict.keys():
        try:
            with open(f"{config_dir}/{filename}.json", "r") as fd:
                loaded[filename] = json.load(fd)
        except Exception as e:
            print(
                f"Unable to find or parse {config_dir}/{filename}.json")
            sys.exit(1)

    # Apply changes from args and write to file
    for filename in cfg_dict.keys():
        # For each element to be replaced in <filename>.json with one of the new args
        for argname in cfg_dict[filename]:
            # Replace loaded arg (old) with arg from argparse (new)
            loaded[filename][argname] = cfg_dict[filename][argname]
            print(f"In {filename}.json, setting '{argname}'={cfg_dict[filename][argname]}")
        
        # Write changes to file
        with open(f"{config_dir}/{filename}.json", "w") as fd:
            json.dump(loaded[filename], fd, indent=4)

if __name__ == "__main__":
    set_cfg()