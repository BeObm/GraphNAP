# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:53:16 2021

@author: Mo
"""
from settings.config_file import *

from  search_space_manager.search_space import *
from search_space_manager.sample_models import *
from search_algo.PCRS import *
from search_space_manager.function_controller import *
from search_space_manager.sample_models import *
from search_algo.write_results import *
from search_algo.get_test_acc import *
from search_algo.importance_ranking import *
from load_data.load_data import *
from datetime import date
import random


now = datetime.now()
import time
dataset_name="PROTEINS"
source='TUDataset'
# source='Planetoid'
# source="Coauthor"
root = f'data/{source}'
dataset_root0=(root,dataset_name,source) 


from settings.config_file import *
 

if __name__ == "__main__":
     add_config("param","type_sampling","random_sampling")
     dataset_root=get_dataset(dataset_root0)
    
     type_task="graph classification"
     timestart = time.time()
     e_search_space= create_e_search_space()

   #-------------- Get few models performance distribution
     
     performance_records = get_performance_distribution(e_search_space, dataset_root)
     
     
     best_model,regressor,predictor_score1, predictor_score2 = get_best_performance_prediction(performance_records,e_search_space,dataset_root)
    
    
    
     facc,stds=get_test_accuracy(best_model, dataset_root)
     # add_config("results","pred_time",prediction_time)
        
     best_txt= f'    ***** Result report for {dataset_root[1]} with sample={config["param"]["n_sample"]} ****** \n -- Regressor = {regressor} | regressor r2 score1 ={round(predictor_score1,2)} and  Predictor r2 score2= :{round(predictor_score2,2)}) \
     \n The best model is : {best_model} | accuracy:{facc} % +/-{round(stds,2)}%  \n \n'
    
     recor_best_file="results/best_model.txt"
     with open(recor_best_file,'a+') as bestmodel:
        bestmodel.write(best_txt)
        
     print(best_txt)   
     # total_time= round(time.time() - timestart,2)
     write_results(dataset_name,round(time.time() - timestart,2),best_model,facc,stds,regressor,predictor_score1,predictor_score2)
     
    