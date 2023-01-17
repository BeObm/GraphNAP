# -*- coding: utf-8 -*-

from settings.config_file import *
 

def create_e_search_space(a=0,b=1):   # a<b

    type_task=config['dataset']["type_task"]
   
    nfcode=int(config["param"]["nfcode"])
    noptioncode=int(config["param"]["noptioncode"])
  

    attention= ["GCNConv","GENConv","SGConv","linear","GraphConv"]

    agregation=['add',"max","mean"]
    activation=["Relu","Elu","linear","Softplus"]
    multi_head= [1]
    hidden_channels =[32]
    normalizer=["GraphNorm","InstanceNorm"]
    dropout = [0.0, 0.2, 0.4, 0.6]
    sp={}  
                                                 
    sp['gnnConv1']=attention
    sp['gnnConv2']= attention
    sp['aggregation1']=agregation
    sp['aggregation2']=agregation
    sp['normalize1'] = normalizer
    sp['normalize2'] = normalizer
    sp['activation1']=activation
    sp['activation2']=activation
    sp['multi_head1']=multi_head
    sp['multi_head2']=multi_head
    sp['hidden_channels1']= hidden_channels
    sp['hidden_channels2']= hidden_channels
    sp['dropout1']= dropout
    sp['dropout2']= dropout
    sp['lr']= [0.01,0.001,0.005,0.0005]
    sp['weight_decay']=[0,0.001,0.0005]
    sp["optimizer"] = ["adam"]
    sp['criterion'] = ['CrossEntropyLoss',"fn_loss"]

    if type_task=='graph classification':
        sp['pooling'] = ["global_add_pool","global_max_pool"]
    # elif type_task=='node classification' or type_task=="link prediction":
    #   sp['normalize1'] =["False","InstanceNorm"]
    #     sp['normalize2'] =["False","InstanceNorm"]
        
    # For quick test the following search space will be used ## MUwech
    total_choices=0    
    t1=1
    max_option=0
    for k,v in sp.items():
        t1=t1*len(v)
        total_choices=total_choices+len(v)
        if len(v)>max_option:
            max_option=len(v)
    add_config("param","max_option",max_option)    
    add_config("param","total_function",len(sp))
    add_config("param","total_choices",total_choices)
    add_config("param","size_sp",t1)
   
  
    print(f'The search space has {len(sp)} functions, a total of {total_choices} choices and {t1} possible GNN models.')
    
    e_search_space,option_decoder = search_space_embeddings(sp,nfcode, noptioncode,a,b)
   
    
    return e_search_space,option_decoder


def create_baseline_search_space(a=0,b=1):   # a<b
    """
    Function to generate architecture description components

    Parameters
    ----------
    nfcode : TYPE   int
        DESCRIPTION.   number of character to encode the type of function in the search space
    noptioncode : TYPE   int
        DESCRIPTION. number of character to encode a choice of a function in the search space
      
    Returns
    ------
    e_search_space : TYPE  dict
        DESCRIPTION.     enbedded search space

    """
    type_task=config['dataset']["type_task"]
   
    nfcode=int(config["param"]["nfcode"])
    noptioncode=int(config["param"]["noptioncode"])
  
    # attention= ["GATConv","GCNConv",'GENConv','GraphUNet',"HypergraphConv","GraphConv","GATConv","GCNConv",
    #            'SuperGATConv',"SAGEConv","ChebConv","ResGatedGraphConv","MFConv","SGConv","ARMAConv","TAGConv","GATv2Conv",
    #            "FeaStConv","PDNConv","EGConv","ClusterGCNConv","LEConv"] 
 
  
    attention= ["GCNConv","GATConv","linear","gat_sym"]
    # attention= ["GCNConv","GENConv","linear","SGConv",'LEConv','ClusterGCNConv', ]

    agregation=['add',"max","mean"] 
    activation=["elu","leaky_relu","linear","relu","relu6","sigmoid","softplus","tanh"]
    multi_head= [1,2,3,4]
    hidden_channels =[8,16,32,64]
    
    dropout = [0.2]    
    sp={}  
           
                                     
    sp['gnnConv1'] = attention
    sp['gnnConv2'] = attention
    sp['aggregation1'] = agregation 
    sp['aggregation2'] = agregation
    sp['activation1'] = activation
    sp['activation2'] = activation
    sp['multi_head1'] = multi_head
    sp['multi_head2'] = multi_head
    sp['hidden_channels1']= hidden_channels
    sp['hidden_channels2']= hidden_channels
   
    # sp['normalize2']=normalizer 
    sp['dropout1']= dropout
    sp['dropout2']= dropout
    # sp['dropout2']= dropout
    sp['lr']= [1e-2, 1e-3, 1e-4, 5e-3, 5e-4]
    sp['weight_decay']=[1e-3, 1e-4, 1e-5, 5e-5, 5e-4]
    
    if type_task=='graph classification':
       
        sp['criterion']= ["fn_loss"] #,"MultiMarginLoss",""fn_loss" 
        sp['pooling']=["global_add_pool"]
        sp["optimizer"] = ["adam"]#,"sgd"
        sp['normalize1'] =["False", "GraphNorm"]
        sp['normalize2'] =["False", "GraphNorm"]
    elif type_task=='node classification' or type_task=="link prediction": 
        
        sp['criterion']= ["fn_loss"] #,"MultiMarginLoss",""fn_loss"
        sp["optimizer"] = ["adam"]#,"sgd"]
        sp['normalize1'] =["False"]
        sp['normalize2'] =["False"]
        
    # For quick test the following search space will be used ## MUwech
    total_choices=0    
    t1=1
    max_option=0
    for k,v in sp.items():
        t1=t1*len(v)
        total_choices=total_choices+len(v)
        if len(v)>max_option:
            max_option=len(v)
    add_config("param","max_option",max_option)    
    add_config("param","total_function",len(sp))
    add_config("param","total_choices",total_choices)
    add_config("param","size_sp",t1)
   
  
    print(f'The search space has {len(sp)} functions, a total of {total_choices} choices and {t1} possible GNN models.')
    
    e_search_space,option_decoder = search_space_embeddings(sp,nfcode, noptioncode,a,b)
   
    
    return e_search_space,option_decoder


 

def search_space_embeddings(sp,nfcode, noptioncode,a,b):
 
    i=0
    
    embeddings_dict={}
    option_decoder={}      # cle= option code, valeur = option
    fcode_list=[]   # list to check duplicate code in function code
         #  liste to check duplicate in option code
    set_seed()
    for function,options_list in sp.items():  
        embeddings_dict[function]={}
        option_code_list = []
        if function in ["gnnConv2","activation2","multi_head2","aggregation2","normalize2",'dropout2']:
            for option in options_list:
                option_code =i 
                i+=1               
                embeddings_dict[function][option]=(option_code, embeddings_dict[f"{function[:-1]}1"][option][1])  
                          
                option_decoder[option_code]=option
                                                
        else:           
            if config["param"]["encoding_method"] =="embedding":
                fcode=[random.randint(a, b) for num in range(0, nfcode)]
               
               # verifier si une autre fonction na pas le meme code avant de valider le code 
                while fcode in fcode_list:
                    fcode=[random.randint(a, b) for num in range(0, nfcode)]
                fcode_list.append(fcode)
                    
                for option in options_list:
                    
                    option_code =i
                    option_encoding=fcode +[random.randint(a, b) for num in range(0, noptioncode)]
                    i+=1
                    while option_encoding in option_code_list:
                        print("option encoding alredy exist")
                        option_encoding = fcode + [random.randint(a, b) for num in range(0, noptioncode)]
                    option_code_list.append(option_encoding)
                    
                    embeddings_dict[function][option]=(option_code,option_encoding) 
                    
                     # set decoder dict value for the current option
                    option_decoder[option_code]=option
            else:
                for option in  options_list:
                   option_code =i 
                   i+=1
                   option_encoding= sp[function].index(option)
                   embeddings_dict[function][option]=(option_code,option_encoding)
                   option_decoder[option_code]=option
                   
    
    return embeddings_dict,option_decoder

