# -*- coding: utf-8 -*-


import torch
import random
import numpy as np
from configparser import ConfigParser
import os.path as osp
import os
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 4
num_seed = 12345
config = ConfigParser()
Batch_Size = 32

# 
RunCode = dates = datetime.now().strftime("%d-%m_%Hh%M")


def set_seed(num_seed=num_seed):
    # os.CUBLAS_WORKSPACE_CONFIG="4096:8"
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(num_seed)
    torch.cuda.manual_seed_all(num_seed)
    np.random.seed(num_seed)
    random.seed(num_seed)


# =========== First level of  runing configurations  =================>

project_root_dir = os.path.abspath(os.getcwd())

type_task = "graph_classification"  # it could be "graph classification", "link prediction",node classification
dataset_name = "IMDB-BINARY"  # Citeseer,


# Second  level of  running configurations
def create_config_file():
    configs_folder = osp.join(project_root_dir, f'results/{type_task}/{dataset_name}/{RunCode}')
    os.makedirs(configs_folder, exist_ok=True)
    config_filename = f"{configs_folder}/ConfigFile_{RunCode}.ini"

    # No neeed to fill dataset information twice
    config["dataset"] = {

        "dataset_name": dataset_name,  # Citeseer,
        'type_task': type_task,  # it could be "graph classification", "link prediction",node classification
        "dataset_root": f"{project_root_dir}/data/{type_task}",
        "shufle_dataset":False
    }

    # fill other configuration information
    config["param"] = {
        "project_dir": project_root_dir,
        'config_filename': config_filename,
        "run_code": RunCode,
        "budget": 800,
        "k": 150,
        "z_sample": 1,  # Number of time  sampled models are trained before we report their performance
        "z_topk": 1,
        "z_final": 5,
        "nfcode": 56,  # number of digit for each function code when using embedding method
        "noptioncode": 8,
        "sample_model_epochs":100,
        "topk_model_epochs": 100,
        "best_model_epochs": 100,
        'search_metric':"balanced_accuracy_score",    #matthews_corr_coef, balanced_accuracy_score, accuracy_score
        "encoding_method": "one_hot",  # ={one_hot, embedding,index_embedding}
        "type_sampling": "controlled_stratified_sampling",  # random_sampling, uniform_sampling, controlled_stratified_sampling
        "predictor_dataset_type": "graph",
        "feature_size_choice": "total_choices",  # total_functions total_choices  # for one hot encoding using graph dataset for predictor, use"total choices
        'type_input_graph': "directed",
        "use_paralell": "no",
        "learning_type": "supervised",
        "predict_sample": 500000,
        "batch_sample": 10000
    }

    config["predictor"] = {

        "dim": 128,
        "drop_out": 0.2,
        "lr": 0.001,
        "wd": 0.0005,
        "num_epoch": 500,
        "comit_test": "yes"

    }

    config["time"] = {
        "distribution_time": 00,
        "sampling_time": 00
    }

    with open(config_filename, "w") as file:
        config.write(file)


def add_config(section_, key_, value_, ):
    if section_ not in list(config.sections()):
        config.add_section(section_)
    config[section_][key_] = str(value_)
    filename = config["param"]["config_filename"]
    with open(filename, "w") as conf:
        config.write(conf)


def create_paths():
    # Create here path for recording model performance distribution
    result_folder = osp.join(project_root_dir, f'results/{type_task}/{dataset_name}/{RunCode}')
    os.makedirs(result_folder, exist_ok=True)
    add_config("path", "performance_distribution_folder", result_folder)
    add_config("path", "best_model_folder", result_folder)

    # Create here path for recording details about the result
    result_detail_folder = osp.join(project_root_dir, f'results/result_details/{type_task}')
    os.makedirs(result_detail_folder, exist_ok=True)
    add_config("path", "result_detail_folder", result_detail_folder)

    # Create here path for saving plots
    plots_folder = osp.join(result_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    add_config("path", "plots_folder", plots_folder)

    # create here path for saving predictor results
    predictor_results_folder = osp.join(result_folder, "predictor_training_data")
    os.makedirs(predictor_results_folder, exist_ok=True)
    add_config("path", "predictor_results_folder", predictor_results_folder)

    add_config("path", "predictor_dataset_folder", predictor_results_folder)

    add_config("path", "predictor_weight_path", result_folder)
