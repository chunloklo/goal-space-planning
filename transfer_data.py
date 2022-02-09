from json import load
import os
import sys

from experiment_utils.data_io.io.zodb_io import save_config_and_data_zodb
sys.path.append(os.getcwd())

from experiment_utils.data_io.configs import load_all_db_configs_and_keys, IdLinkedConfig, load_data_from_config_id_zodb, save_data_zodb
from  experiment_utils.analysis_common.configs import SweepInfo
from tqdm import tqdm

from_db_folder = 'results_dbs/new_graze_ideal'
to_db_folder = 'results_dbs/graze_ideal'

configs = load_all_db_configs_and_keys(from_db_folder)

for config in tqdm(configs):
    data = load_data_from_config_id_zodb(from_db_folder, config.get_id())
    # save_data_zodb(config, data)
    save_config_and_data_zodb(config.get_id(), config, data, to_db_folder)

# print(configs)

# print(config_id_tuples[0])

# sweepInfo = SweepInfo(configs)
# sweepInfo.add_filter('alpha', 0.3)
# print(sweepInfo.diff_list)
# print(len(sweepInfo.filtered_configs))

# print(sweepInfo.filtered_configs[0].get_id())
