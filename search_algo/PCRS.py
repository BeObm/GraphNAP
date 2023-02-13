 # -*- coding: utf-8 -*-


import time
import pandas as pd
import statistics as stat
import copy
from collections import defaultdict
from search_algo.utils import *
from torch_geometric.data import Data
from search_algo.predictor_model import *
from search_space_manager.sample_models import *
from load_data.load_data import load_dataset
from search_space_manager.search_space import *
from search_space_manager.map_functions import *
import torch.nn as nn
from GNN_models.node_classification import *
from GNN_models.graph_classification import *
from settings.config_file import *
from collections import OrderedDict
import importlib

set_seed()

def get_performance_distributions(e_search_space,dataset):  # get performance distribution of s*n models (n = search space size)
  
    z_sample=int(config["param"]["z_sample"])

    type_task =config["dataset"]["type_task"]
    epochs =int(config["param"]["sample_model_epochs"])
    n_sample =int(config["param"]["N"]) 
    search_metric =config["param"]["search_metric"]

    timestart = time.time()
    print(f' \n Getting {search_metric}  of  {n_sample} models ...\n')

    gcn,train_model,test_model=get_train(type_task)
    nummodel=1
    best_performance=0
    model_list = sample_models(n_sample, e_search_space)
    edge_index = get_edge_index(model_list[0])
    # print("example of model_config", model_list[0])
    predictor_dataset=defaultdict(list)
    graph_list=[]

    train_loader, val_loader, test_loader = load_dataset(dataset)
    for no, submodel in tqdm(enumerate(model_list)):

            set_seed()
            torch.cuda.empty_cache()

            model, criterion, optimizer =get_model_instance(submodel, dataset,gcn)
            
            performance_record=[]

            for i in range(z_sample):
                set_seed()
                for epoch in range(epochs):                                            
                   train_model(model,train_loader, criterion, optimizer)
                   performance_score= test_model(model, val_loader)
                print(f"this is the variable {search_metric}")
                performance_record.append(performance_score[search_metric])
            model_performance = round(stat.mean(performance_record),8)
            
           
            if model_performance >= best_performance:
                best_performance=model_performance
                best_sample=copy.deepcopy(submodel)
                best_sample[search_metric]=best_performance
                print(f'-->  {search_metric} = {model_performance}  ===++ Actual Best Performance')
                  
            else :
                  print(f'--> {search_metric} = {model_performance}')
            print([submodel[opt][0] for opt in submodel.keys()])
            

            #  transform model configuration into graph data
            if (config["param"]["predictor_dataset_type"])=="graph":
                # edge_index=get_edge_index(model_list[0])
                x = get_nodes_features(submodel,e_search_space)
                y=np.array(model_performance)
                y=torch.tensor(y,dtype=torch.float32).view(-1,1)
                graphdata=Data(x=x,edge_index =edge_index,y=y,num_nodes=x.shape[0],model_config_choices = deepcopy(submodel))
                graph_list.append(graphdata)                                   
                torch.save(graphdata,f"{config['path']['predictor_dataset_folder']}/graph{nummodel}_{x.shape[1]}Feats.pt")                   
                
            elif (config["param"]["predictor_dataset_type"])=="table":
                for function,option in submodel.items():
                    if config["param"]["encoding_method"] =="one_hot":
                       predictor_dataset[function].append(option[0])
                    elif config["param"]["encoding_method"] =="embedding":
                       predictor_dataset[function].append(option[2])
                predictor_dataset[search_metric].append(model_performance)
            nummodel+=1      
                 
                             
                
    sample_time= round(time.time() - timestart,2)
    add_config("time","distribution_time",sample_time)
    add_config("results",f"best_{search_metric}",best_performance)
    
    if (config["param"]["predictor_dataset_type"])=="graph":
                 
         return  config['path']['predictor_dataset_folder'], best_sample     
   
    if (config["param"]["predictor_dataset_type"])=="table":
        df =pd.DataFrame.from_dict(predictor_dataset,orient="columns")   
        dataset_file=f'{config["path"]["predictor_dataset_folder"]}/{config["dataset"]["dataset_name"]}-{config["param"]["budget"]} samples.csv'
        df.to_csv(dataset_file)                  
        return dataset_file, best_sample   
 
    
def get_best_model(topk_list,option_decoder,dataset):

    torch.cuda.empty_cache()
    search_metric = config["param"]["search_metric"]
    best_loss_param_path =f"{config['path']['performance_distribution_folder']}/best_dist_param.pth"
    set_seed()
    encoding_method =config["param"]["encoding_method"]
    type_task =config["dataset"]["type_task"]
    z_topk= int(config["param"]["z_topk"])
    epochs= int(config["param"]["topk_model_epochs"])
    n_sample =int(config["param"]["N"])
    type_sampling= config["param"]["type_sampling"]
    start_time = time.time()
    
    task_model,train_model,test_model=get_train(type_task)
    print("Training top-k models ...")
    try:
        Y = 0
        for filename in glob.glob(config["path"]["predictor_dataset_folder"] + '/*'):
            data = torch.load(filename)
            data.y = data.y.view(-1, 1)
            if Y < data.y.item():
                Y = data.y.item()
                submodel = data.model_config_choices
            max_performace = Y

            bestmodel = copy.deepcopy(submodel)

            for k, v in bestmodel.items():
                if k != search_metric:
                    bestmodel[k] = v[0]
    except:
        max_performace =0

    for idx,row in tqdm(topk_list.iterrows()):
        dict_model={}   #
        
        if (config["param"]["predictor_dataset_type"])=="graph":
            for choice in row["model_config"]:
                dict_model[choice[0]]= option_decoder[choice[1]]
    
        elif (config["param"]["predictor_dataset_type"])=="table":
            for function in topk_list.columns: 
                if function !=search_metric:
                    if config["param"]["encoding_method"] =="one_hot":
                      dict_model[function]=row[function]
                    elif config["param"]["encoding_method"] =="embedding":
                      dict_model[function]=option_decoder[row[function]]
        
        set_seed()
        model, criterion, optimizer =get_model_instance2(dict_model, dataset,task_model)
        
        train_loader,val_loader,test_loader = load_dataset(dataset)
        try:
            model.load_state_dict(best_loss_param_path)
           
        except:
             model, criterion, optimizer =get_model_instance2(dict_model, dataset,task_model)
        
        performance_list=[]

        for i in range(z_topk):
            best_loss=999
            set_seed()
            for epoch in range(epochs):                                            
               loss = train_model(model,train_loader, criterion, optimizer)
               if loss < best_loss:
                  best_loss = loss 
                  best_model = copy.deepcopy(model)
                  torch.save(model.state_dict(),best_loss_param_path )
                  
            #       # torch.save(model.state_dict(), f'{config["path"]["best_model_folder"]}/temp_model_dict.pth')  
            best_model, criterion, optimizer = get_model_instance2(dict_model, dataset,task_model)
            best_model.load_state_dict(torch.load(best_loss_param_path))          
                    
            performance= test_model(best_model, val_loader)
            performance_list.append(performance[search_metric])

        performance = round(stat.mean(performance_list),8)


                 
        if max_performace <= performance :
            max_performace = performance
            bestmodel=copy.deepcopy(dict_model)
    get_best_model_time = round(time.time() - start_time,2)
    add_config("time","get_best_model_time",get_best_model_time)

    return bestmodel       



   
def get_train(type_task):

    task_model_obj = importlib.import_module(f"GNN_models.{type_task}")
    gcn = getattr(task_model_obj,"GNN_Model")
    train_model = getattr(task_model_obj, "train_function")
    test_model = getattr(task_model_obj, "test_function")
    return gcn,train_model,test_model




def get_model_instance(submodel,dataset,GCN):

    
    
    set_seed()
    
    type_task =config["dataset"]["type_task"]
    param_dict={}
    # dist_dict_encoded, sp_dict_decoder = encode_sp(submodel, 'unique_option')    # pour enregitrer just un model    dans le dataset                      
    param_dict['aggregation1']= submodel['aggregation1'][0]
    param_dict['aggregation2']= submodel['aggregation2'][0]
    param_dict["normalize1"] = map_normalization(submodel["normalize1"][0])
    param_dict["normalize2"] = map_normalization(submodel["normalize2"][0])
    param_dict["dropout1"] = submodel["dropout1"][0]                
    param_dict["dropout2"] = submodel["dropout2"][0]
    param_dict["multi_head1"] = submodel["multi_head1"][0]
    param_dict["multi_head2"] = submodel["multi_head2"][0]
    param_dict['activation1']=map_activation_function(submodel['activation1'][0])
    param_dict['activation2']=map_activation_function(submodel['activation2'][0])
    if type_task=='graph_classification':
        param_dict["global_pooling"]=map_pooling(submodel['pooling'][0])
    param_dict['type_task'] =type_task
    param_dict["gnnConv1"]=map_gnn_model(submodel['gnnConv1'][0])
    param_dict["gnnConv2"]=map_gnn_model(submodel['gnnConv2'][0])
    param_dict["hidden_channels1"]=submodel['hidden_channels1'][0] 
    param_dict["hidden_channels2"]=submodel['hidden_channels2'][0] 
    param_dict["dataset"]=dataset 
    model=GCN(param_dict)
    if torch.cuda.device_count() > 1 and config["param"]["use_paralell"]=="yes" :
        print("Multi GPU detected, Parallel computation activated!")
        model = nn.DataParallel(model)
        
    model.to(device)
    
    criterion = map_criterion(submodel['criterion'][0])
    optimizer=map_optimizer(submodel['optimizer'][0] , model, submodel['lr'][0], submodel['weight_decay'][0]) 
              
    return model,criterion,optimizer

def get_model_instance2(submodel,dataset,GCN):

    set_seed()
    type_task =config["dataset"]["type_task"]
    param_dict={}
    # dist_dict_encoded, sp_dict_decoder = encode_sp(submodel, 'unique_option')    # pour enregitrer just un model    dans le dataset                      
    param_dict['aggregation1']= submodel['aggregation1']
    param_dict['aggregation2']= submodel['aggregation2']
    param_dict["normalize1"] = map_normalization(submodel["normalize1"])
    param_dict["normalize2"] = map_normalization(submodel["normalize2"])
    param_dict["dropout1"] = submodel["dropout1"]                
    param_dict["dropout2"] = submodel["dropout2"]
    param_dict["multi_head1"] = submodel["multi_head1"]
    param_dict["multi_head2"] = submodel["multi_head2"]
    param_dict['activation1']=map_activation_function(submodel['activation1'])
    param_dict['activation2']=map_activation_function(submodel['activation2'])
    if type_task=='graph_classification':
        param_dict["global_pooling"]=map_pooling(submodel['pooling'])
    param_dict['type_task'] =type_task
    param_dict["gnnConv1"]=map_gnn_model(submodel['gnnConv1'])
    param_dict["gnnConv2"]=map_gnn_model(submodel['gnnConv2'])
    param_dict["hidden_channels1"]=submodel['hidden_channels1'] 
    param_dict["hidden_channels2"]=submodel['hidden_channels2'] 
    param_dict["dataset"]=dataset 
    model=GCN(param_dict)
    if torch.cuda.device_count() > 1 and config["param"]["use_paralell"]=="yes":
        print("Multi GPU detected, Parallel computation activated!")
        model = nn.DataParallel(model)
    torch.cuda.empty_cache()
    model.to(device)
    
    criterion = map_criterion(submodel['criterion'])
    optimizer=map_optimizer(submodel['optimizer'] , model, submodel['lr'], submodel['weight_decay']) 
              
    return model,criterion,optimizer