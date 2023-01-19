# -*- coding: utf-8 -*-

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from  search_space_manager.search_space import *
from search_space_manager.sample_models import *
from search_algo.PCRS import *
from search_space_manager.sample_models import *
from search_algo.predictor_model import *
from search_algo.write_results import *
from search_algo.get_test_acc import *
from load_data.load_data import *
from search_algo.utils import manage_budget,Generate_time_cost
from datetime import date
import random
import time
from settings.config_file import *


if __name__ == "__main__":
    torch.cuda.empty_cache()
    set_seed()
    create_config_file()
    timestart = time.time()
    create_paths()  
    manage_budget()

    # torch.cuda.empty_cache()
    type_task=config["dataset"]["type_task"]
    dataset_name=config["dataset"]["dataset_name"]
    dataset_root =config["dataset"]["dataset_root"]
    print(f"code running on {dataset_name} dataset")
    
    dataset=get_dataset(type_task,dataset_root,dataset_name)

    e_search_space,option_decoder = create_e_search_space()
    # e_search_space,option_decoder = create_baseline_search_space()

    performance_records_path = get_performance_distributions(e_search_space, dataset)

    TopK_final = get_prediction(performance_records_path,e_search_space)
    best_model= get_best_model(TopK_final,option_decoder,dataset)
    total_search_time = round(time.time() - timestart, 2)
    add_config("time", "total_search_time", total_search_time)
    get_test_performance(best_model,dataset)
    write_results(best_model)  
    # Generate_time_cost()