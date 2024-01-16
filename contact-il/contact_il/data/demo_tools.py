import argparse
import os
from contact_il.data.dict_dataset import DictDataset


parser = argparse.ArgumentParser()
parser.add_argument('dataset_folder', type=str,  help='absolute path to dataset folder after CIL_DATA_DIR')
parser.add_argument('--save_dir', type=str, default=os.environ["CIL_DATA_DIR"],
                    help='Top level directory for dataset.')

args = parser.parse_args()

ds = DictDataset(
    pa_args=args,
    dataset_name="",
    main_dir=os.path.join(args.save_dir, args.dataset_folder),
)

import ipdb; ipdb.set_trace()