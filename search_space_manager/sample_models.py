# -*- coding: utf-8 -*-

import random
import pandas as pd
from itertools import product
from search_space_manager.search_space import *
import inspect 
import time
from search_space_manager.map_functions import map_gnn_model
from tqdm import tqdm
from collections import defaultdict
import itertools
import copy
from sklearn.model_selection import ParameterGrid

from settings.config_file import *
 


def sample_models(n_sample,e_search_space):
    set_seed()
    
    type_sample =config["param"]["type_sampling"]

    if type_sample =="uniform_sampling":
        models_sampled_list = uniform_sample(n_sample, e_search_space)
    elif type_sample =="random_sampling":
        models_sampled_list = random_sample(e_search_space,n_sample)
    elif type_sample == "controlled_stratified_sampling":
          models_sampled_list = controlled_stratified_sample(n_sample,e_search_space)
    else:
        raise ValueError("sampling type incorrect, please chek and try again !")

    return models_sampled_list



        


def random_sample(e_search_space,n_sample=0,predictor=False):   

    timestart = time.time()
    
    set_seed()
    if n_sample!=0:
        
            temp_dict={}
            for k,v in e_search_space.items():
                temp_dict[k]=list(v.keys())
            model_list=[]
            lst=['head','heads','num_heads']
            for i in tqdm(range(n_sample)):       
                model_dict={}
                for name,val in e_search_space.items():  
                    choice= random.choice(list(val.keys()))
                    model_dict[name]=(choice,e_search_space[name][choice][0],e_search_space[name][choice][1])
                    
                 # force the multi head number to be 1 for convolution that does not apply multi-attention operation   
                conv1= map_gnn_model(model_dict["gnnConv1"][0])[0] 
                if len([a for a in lst if a in inspect.getfullargspec(conv1)[0]])==0:                
                    model_dict["multi_head1"]=(1,e_search_space["multi_head1"][1][0],e_search_space["multi_head1"][1][1])
                   
                conv2= map_gnn_model(model_dict["gnnConv2"][0])[0]           
                if len([a for a in lst if a in inspect.getfullargspec(conv2)[0]])==0:
                    
                    model_dict["multi_head2"]=(1,e_search_space["multi_head2"][1][0],e_search_space["multi_head2"][1][1])
                     
                model_list.append(model_dict)
                # if model_dict not in model_list:  # to avoid model's duplicate
                #   model_list.append(model_dict)
                # else:
                #     print("model already sampled, sampling again")
    else:
       print('I am sampling all models from a vast search space, please be patient ...!')
       keys, values = zip(*e_search_space.items())
       model_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
       print("Total sampled models: ",len(model_list))
    if predictor ==True:
        sampling_time = round(time.time() - timestart,2)
        add_config("time","sampling_time",sampling_time)
            
    return model_list




def uniform_sample(n_sample,e_search_space,predictor=False):
    set_seed()
    temp_dict={}
    for k,v in e_search_space.items():
        temp_dict[k]=list(v.keys())
    lst=['head','heads','num_heads']    
    model_list=[]
    while len(model_list)< n_sample:       
        model_dict={}
        for name,val in e_search_space.items():  
            if len(temp_dict[name])==0:
                temp_dict[name]=list(val.keys())   
            choice= random.choice(temp_dict[name])
            model_dict[name]=(choice,e_search_space[name][choice][0],e_search_space[name][choice][1])
            temp_dict[name].remove(choice)
                        
        # force the multi head number to be 1 for convolution that does not apply multi-attention operation   
        conv1= map_gnn_model(model_dict["gnnConv1"][0])[0] 
        if len([a for a in lst if a in inspect.getfullargspec(conv1)[0]])==0:                
            model_dict["multi_head1"]=(1,e_search_space["multi_head1"][1][0],e_search_space["multi_head1"][1][1])
           
        conv2= map_gnn_model(model_dict["gnnConv2"][0])[0]           
        if len([a for a in lst if a in inspect.getfullargspec(conv2)[0]])==0:
            
            model_dict["multi_head2"]=(1,e_search_space["multi_head2"][1][0],e_search_space["multi_head2"][1][1])  
       
        if model_dict not in model_list:  # to avoid model's duplicate
          model_list.append(model_dict)
        else:
             print("model already sampled, sampling again")
         
    # print(f"this is how a sampled model looks like {model_list[:2]}")
    return model_list





def controlled_stratified_sample(n_sample,e_search_space,predictor=False):
    set_seed()
    temp_dict={}
    total_choices=0
    for k,v in e_search_space.items():
        temp_dict[k]=list(v.keys())
        total_choices += len(v)  
      
    s=int(n_sample/total_choices)+1
    model_list=[]
    lst=['head','heads','num_heads']  
    while len(model_list)< s:       
        model_dict={}
        for name,val in e_search_space.items():  
            if len(temp_dict[name])==0:
                temp_dict[name]=list(val.keys())   
            choice= random.choice(temp_dict[name])
            model_dict[name]=(choice,e_search_space[name][choice][0],e_search_space[name][choice][1])
            temp_dict[name].remove(choice)
            
        # force the multi head number to be 1 for convolution that does not apply multi-attention operation   
        conv1= map_gnn_model(model_dict["gnnConv1"][0])[0] 
        if len([a for a in lst if a in inspect.getfullargspec(conv1)[0]])==0:                
            model_dict["multi_head1"]=(1,e_search_space["multi_head1"][1][0],e_search_space["multi_head1"][1][1])
           
        conv2= map_gnn_model(model_dict["gnnConv2"][0])[0]           
        if len([a for a in lst if a in inspect.getfullargspec(conv2)[0]])==0:
            
            model_dict["multi_head2"]=(1,e_search_space["multi_head2"][1][0],e_search_space["multi_head2"][1][1])  
       
        if model_dict not in model_list:  # to avoid model's duplicate
          model_list.append(model_dict)
        else:
             print("model already sampled, sampling again")
         
    model_list2 =[]    
    for function, option in e_search_space.items():
        for component in option.keys():
            for submodel in model_list:
                model_temp = copy.deepcopy(submodel)
                model_temp[function]=(component,option[component][0],option[component][1])
                model_list2.append(model_temp)
                
                
    # print(model_list[:2])      
    # print("|n")      
    # print(model_list2[:2])   
    return random.sample(model_list2,n_sample)










def random_sample_backup(e_search_space,n_sample=0,predictor=False):   

    timestart = time.time()
   
    
    
    if n_sample!=0:
        model_dict=defaultdict(list)
        samples_list=[]
        for i in tqdm(range(n_sample)):             
            for function,options in e_search_space.items():    
                choice=random.choice([option for option in options.keys()])
                model_dict[function].append((choice,e_search_space[function][choice][0],e_search_space[function][choice][1]))
                           
        df =pd.DataFrame.from_dict(model_dict)   
        df=df.drop_duplicates()
        while n_sample>(len(df)+10):
           for i in tqdm(range(int(n_sample/4))):       
           
             for function,options in e_search_space.items():  
            
                choice=random.choice([option for option in options.keys()])
                model_dict[function].append((choice,e_search_space[function][choice][0],e_search_space[function][choice][1]))
  
        
  
           df =pd.DataFrame.from_dict(model_dict)   
           df=df.drop_duplicates()
    
        df =pd.DataFrame.from_dict(model_dict)   
        df=df.drop_duplicates()
        df=df[:n_sample]
        samples_list=df.to_dict('records')
        print("lenght of sampled list: ",len(samples_list))
    else:
       print('I am sampling all models from a vast search space, please be patient ...!')
       keys, values = zip(*e_search_space.items())
       samples_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
       print("Total sampled models: ",len(samples_list))
    if predictor ==True:
        sampling_time = round(time.time() - timestart,2)
        add_config("time","sampling_time",sampling_time)
            
    return samples_list