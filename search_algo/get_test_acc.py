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





def get_test_performance(submodel, dataset):

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
    test_performances_record = defaultdict(list)
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

        performances= test_model(best_model, val_loader)  # test_loader

        for k,v in performances.items():
            test_performances_record[k].append(v)
    test_performances_records={}
    for metric, performance in test_performances_record.items():
        test_performances_records[metric]=round(stat.mean(performance),8)
        test_performances_records[f"{metric}_std"] = round(np.std(performance, dtype=np.float64), 8)
        add_config("results", metric, test_performances_records[metric])
        add_config("results", f"{metric}_std", test_performances_records[f"{metric}_std"])

    return test_performances_record
