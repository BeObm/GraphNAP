# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:53:16 2021

@author: Mo
"""


import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from  search_space_manager.search_space import *
from search_space_manager.sample_models import *
from search_algo.PCRS import get_performance_distributions
from search_algo.predictor_model import *
from search_algo.write_results import *
from search_algo.get_test_acc import *
from load_data.load_data import *
from search_algo.utils import manage_budget, Generate_time_cost
from datetime import date
import random
import time
from settings.config_file import *


if __name__ == "__main__":
    torch.cuda.empty_cache()
    set_seed()
    create_config_file()
   
    create_paths()  
    manage_budget()

    # torch.cuda.empty_cache()
    
    dataset_name=config["dataset"]["dataset_name"]
    dataset_source=config["dataset"]["dataset_source"]
    dataset_root =config["dataset"]["dataset_root"]
    datasetRoot=(dataset_root,dataset_name,dataset_source) 
    print(f"code running on {dataset_name} dataset")
    
    dataset=get_dataset(datasetRoot)
     
    e_search_space,option_decoder = create_e_search_space()
    # e_search_space,option_decoder = create_baseline_search_space()
    total_search_timestart = time.time()

    performance_records,bs = get_performance_distributions(e_search_space, dataset)
    
    # performance_records,bs ="D:/PHD/Codes/AutoML/GraphNAP/results/graph classification/PROTEINS/15-07_10h24/predictor_training_data/PROTEINS-100 samples.csv",{}
        
    TopK_final = get_prediction(performance_records,e_search_space)
    
    best_model= get_best_model(TopK_final,option_decoder,dataset,bs)
    test_acc,test_std = get_test_accuracy(best_model,dataset)
    
    total_search_time = round(time.time() - total_search_timestart,2)
    add_config("time","total_search_time",total_search_time)
    
    write_results(best_model)  
    Generate_time_cost()