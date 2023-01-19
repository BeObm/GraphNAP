# -*- coding: utf-8 -*-


from settings.config_file import *
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve,matthews_corrcoef, balanced_accuracy_score,accuracy_score




def set_seed(num_seed=num_seed):
    # os.CUBLAS_WORKSPACE_CONFIG="4096:8"
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(num_seed)
    torch.cuda.manual_seed_all(num_seed)
    np.random.seed(num_seed)
    random.seed(num_seed)


def aucPerformance(y_true, y_pred):
    y_true = y_true.flatten().cpu()
    y_pred = y_pred.flatten().cpu()
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    return roc_auc*100, auc_pr*100


def mcc_score(y_true, y_pred):
    y_true = y_true.flatten().cpu()
    y_pred = y_pred.flatten().cpu()
    mcc = matthews_corrcoef(y_true, y_pred)

    return mcc

def balanced_acc_score(y_true, y_pred):
    y_true = y_true.flatten().cpu()
    y_pred = y_pred.flatten().cpu()
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return balanced_acc

def acc_score(y_true, y_pred):
    y_true = y_true.flatten().cpu()
    y_pred = y_pred.flatten().cpu()
    acc_score = accuracy_score(y_true, y_pred)
    return acc_score

def build_feature(function,option,num_function,e_search_space): # creer les feature d un graph provenant d un submodel
    type_encoding= config["param"]["encoding_method"] 
    total_choices =int(config["param"]["total_choices"])
    total_function =int(config["param"]["total_function"])
    max_option =int(config["param"]["max_option"])
    
    if type_encoding=="one_hot":
        d= np.zeros((total_choices), dtype=int)
        # print("option===",option)    
        d[option[1]]=1
    elif type_encoding =="embedding":
           d=option[2]
    elif type_encoding =="index_embedding":
           if config["param"]["feature_size_choice"] =="total_functions":
              d= np.zeros((total_function), dtype=int)
            
              d[num_function]= list(e_search_space[function]).index(option[0])+1
           else:               
              d= np.zeros((total_choices), dtype=int)
              d[option[1]]=list(e_search_space[function]).index(option[0])+1
    elif  type_encoding=="OneHot":
         d= np.zeros((max_option), dtype=int)
         pos=2
         
    return d



def get_nodes_features(model_config,e_search_space):
       
        nodes_features_list=[]
        model_config_choices=[]
        num_function=0
        for function,option in model_config.items():
             # print("function===",function)
             feat = build_feature(function,option,num_function,e_search_space) 
             num_function+=1
             nodes_features_list.append(feat)
             model_config_choices.append((function,option[1]))    # (AV ANT) quand j enleve la conversion en str cela provoque plus tard des erreurs incomprises au niveau du batching
        x=np.array(nodes_features_list) 
        x = torch.tensor(x,dtype=torch.float32)
        
        return x
    
def get_edge_index(model_config):
    edge_dict={} 
    node_idx={}
    #  variable to controll the random sampling to make sure options are distribute uniformly
    idx=0
    for functions,options in model_config.items():
        node_idx[functions]=idx
        idx+=1
    
    edge_dict['gnnConv1']=["normalize1",'dropout1','activation1']
    edge_dict['aggregation1']=["gnnConv1"]
    edge_dict['multi_head1']=["gnnConv1"]
    edge_dict['hidden_channels1']= ["gnnConv1"]
    edge_dict['normalize1']=["dropout1",'activation1']
    edge_dict['dropout1']=["activation1"]
    edge_dict['activation1']=["gnnConv2"]
   
    edge_dict['gnnConv2']= ["normalize2",'dropout2','activation2']
    edge_dict['aggregation2']=["gnnConv2"] 
    edge_dict['multi_head2']=["gnnConv2"]
    edge_dict['hidden_channels2']= ["gnnConv2"]
    edge_dict['normalize2']=["dropout2",'activation2']
    edge_dict['dropout2']=["activation2"]
   
    if config["dataset"]['type_task']=="graph_classification":
        edge_dict['activation2']= ["pooling"]
        edge_dict['pooling']=['criterion']
    else:
         edge_dict['activation2']= ["criterion"]       
    edge_dict['lr']= ["criterion","weight_decay"]
    edge_dict['weight_decay']=["criterion","lr"]
    edge_dict["criterion"]=["optimizer"]
    edge_dict["optimizer"]=[]
   
    source=[]
    target=[]
    edge_index=[]
    for function,options in model_config.items():
            # source.append(node_idx[function])
            # target.append(node_idx[function])
             
            if config['param']['type_input_graph']=="undirected":
                for function2,options2 in model_config.items():
                    source.append(node_idx[function])
                    target.append(node_idx[function2])
                    # a=node_idx[function]
                    # b=node_idx[function2]
                    # print(f"Edge between {a} and {b}")
            else:
                for elt in edge_dict[function]:
                    source.append(node_idx[function])
                    target.append(node_idx[elt])
                    # a=node_idx[function]
                    # b=node_idx[elt]
                    # print(f"Edge between {a} and {b}")
        
    edge_index.append(source)  
    edge_index.append(target)
    edge_index=np.array(edge_index)
    edge_index=torch.tensor(edge_index,dtype=torch.long)
    return edge_index
    


def manage_budget():
     budget =int(config["param"]["budget"]) 
     k= int(config["param"]["k"]) 
     z_sample= int(config["param"]["z_sample"]) 
   
     z_topk= int(config["param"]["z_topk"]) 
     z_final= int(config["param"]["z_final"]) 
     
     n= int((budget-(k*z_topk)-z_final)/z_sample)
     if n<=0:
         print("Configuration error, Please change budget realated parameters")
         raise SystemExit

     else:
        add_config("param","n",n) 
       
   

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    return g


def load_data(train_dataset,batch_size,num_workers,worker_init_fn,generator):
    
   loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator)

   return loader


def Generate_time_cost():
    dataset_construction = float(config["time"]["distribution_time"])
    predictor_training   = float(config["time"]["predictor_training_time"])
    gnn_encoding         = float(config["time"]["sampling_time"])
    topk_gnn_prediction  = float(config["time"]["pred_time"])
    topk_training        = float(config["time"]["best_acc_time"])
    total=float(config["time"]["total_search_time"])

    # Make a random dataset:
    height = [dataset_construction,predictor_training,gnn_encoding+topk_gnn_prediction,topk_training]
    bars = ('Predictor training dataset construction','predictor training', 'top-k gnn prediction', 'top k gnn training')
    y_pos = np.arange(len(bars))
    fig,ax = plt.subplots()
    # Create bars
    plt.barh(y_pos, height)
    plt.title("Running time details on {dataset_name} dataset")
    plt.ylabel("running time(seconds)")
    # Create names on the x-axis
    plt.xticks(y_pos, bars)
    plt.grid()
    plt.show()
    fig.savefig(f'{config["path"]["plots_folder"]}/{dataset_name}_timeCost_details_bar.pdf',bbox_inches="tight")
     
   
    # explosion
    fig, ax = plt.subplots(figsize=(25,10), subplot_kw=dict(aspect="equal"))

    # Pie Chart
    plt.pie(height, labels=bars,
            autopct='%1.1f%%', pctdistance=0.85)

    # draw circle
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig = plt.gcf()

    # Adding Circle in Pie chart
    fig.gca().add_artist(centre_circle)

    # Adding Title of chart
    plt.title("Running time details on {dataset_name} dataset")

    # Displaing Chart
    plt.show()
    fig.savefig(f'{config["path"]["plots_folder"]}/{dataset_name}_timeCost_details_pie.pdf',bbox_inches="tight")
    
    
   
    bars = ('GraphNAS','RS', 'GAS', 'Auto-GNAS',"GraphNAP")
    if config["dataset"]["dataset_name"]== "Cora":
         height = [12960,12240,11520,3240,total]
    elif  config["dataset"]["dataset_name"]== "Citeseer":
         height = [13320,13248,13680,4140,total]
    elif  config["dataset"]["dataset_name"]== "Pubmed":
          height = [18360,18360,16560,5760,total]  
          
          
    y_pos = np.arange(len(bars))
    fig,ax = plt.subplots()
    # Create bars
    plt.bar(y_pos, height)
    plt.title("Running time Comparison on {dataset_name} dataset")
    plt.ylabel("running time(seconds)")
    # Create names on the x-axis
    plt.xticks(y_pos, bars)
    plt.grid()
    plt.show()
    fig.savefig(f'{config["path"]["plots_folder"]}/{dataset_name}_timeCost_comparison.pdf',bbox_inches="tight")
     
