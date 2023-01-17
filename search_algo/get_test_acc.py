# -*- coding: utf-8 -*-

# import os

import statistics as stat
from search_space_manager.sample_models import *
from load_data.load_data import load_dataset
from search_space_manager.search_space import *
from search_space_manager.map_functions import *
from search_algo.PCRS import *
from GNN_models.node_classification import *
from GNN_models.graph_classification import *





def get_test_accuracy(submodel, dataset):

    z_final= int(config["param"]["z_final"])
    type_task =config["dataset"]["type_task"]
    epochs= int(config["param"]["best_model_epochs"])   
    best_loss_param_path =f"{config['path']['performance_distribution_folder']}/best_dist_param.pth"

  
    type_task =config["dataset"]["type_task"]
   
    n_sample =int(config["param"]["N"]) 
    
    timestart = time.time()   # to record the total time to get the performance distribution set
    print(f' \n Getting performance accuracy  of  {n_sample} models ...\n')
  
    set_seed()
    gcn,train_model,test_model=get_train(type_task)   
    
    train_loader,val_loader,test_loader = load_dataset(dataset)
   
      
    model, criterion, optimizer =get_model_instance2(submodel, dataset,gcn)
    auc_roc_list=[]
    auc_pr_list = []

    set_seed()
    for i in range(z_final):                
        best_loss =999
        for epoch in range(epochs):                                            
           loss = train_model(model,train_loader, criterion, optimizer)  
                
           if loss <= best_loss:
                  best_loss = loss
                  torch.save(model.state_dict(),best_loss_param_path )
                  
        # # torch.save(model.state_dict(), f'{config["path"]["best_model_folder"]}/temp_model_dict.pth')  
        best_model, criterion, optimizer =get_model_instance2(submodel, dataset,gcn)
        best_model.load_state_dict(torch.load(best_loss_param_path))

        auc_roc, auc_pr= test_model(best_model, val_loader)  # test_loader

        auc_pr_list.append(auc_pr)
        auc_roc_list.append(auc_roc)
    auc_pr = round(stat.mean(auc_pr_list),2)
    auc_roc = round(stat.mean(auc_roc_list), 2)
    auc_pr_std=round(np.std(auc_pr_list,dtype = np.float64),2)
    auc_roc_std = round(np.std(auc_roc_list, dtype=np.float64), 2)


    add_config("results","best_auc_pr",auc_pr)
    add_config("results", "best_auc_roc", auc_roc)
    add_config("results","std_auc_pr",auc_pr_std)
    add_config("results", "std_auc_roc", auc_roc_std)
    
    return auc_pr,auc_pr_std
         