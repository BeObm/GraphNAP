# -*- coding: utf-8 -*-

# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
import time
from tqdm import tqdm
import torch.nn as nn
import math
import scipy.stats as stats
from search_algo.utils import *
from search_space_manager.map_functions import map_activation_function
from torch_geometric.nn import MessagePassing
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from search_space_manager.search_space  import *
from search_space_manager.sample_models import *
from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch_geometric.nn import global_add_pool #global_mean_pool, global_max_pool,
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import glob
from torch_geometric.nn.norm import GraphNorm
from sklearn.linear_model import SGDRegressor,LassoCV
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv,SAGEConv,GATConv,LEConv,GENConv,GeneralConv,TransformerConv
from torch_geometric.nn.norm import GraphNorm, InstanceNorm,BatchNorm

from settings.config_file import *

set_seed()


def get_prediction(performance_records_path,e_search_space):
    if (config["param"]["predictor_dataset_type"])=="graph":
       TopK_final= get_prediction_from_graph(e_search_space)
    elif (config["param"]["predictor_dataset_type"])=="table":
       TopK_final= get_prediction_from_table(performance_records_path,e_search_space)
    return TopK_final
 
    


class Predictor(MessagePassing):
    def __init__(self, in_channels, dim, out_channels,drop_out):
        super(Predictor, self).__init__()
#         self.embed_edges = Linear(self.edge_attr_size, self.hidden_channels) 
        # print("in channels dim =",in_channels)
        self.conv1 = GraphConv(in_channels, dim, aggr="add")
        
        self.conv2 = GraphConv(dim, dim,aggr="add")
        self.drop_out=drop_out
        # self.normalize = InstanceNorm(dim)
        self.graphnorm=GraphNorm(dim)
        self.linear=Linear(dim,64)
        self.linear2=Linear(64,out_channels)
      
        
    def forward(self, x, edge_index, batch):
        # x, edge_index, batch= data.x, data.edge_index, data.batch
        
       
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.drop_out,training=self.training)
        
        x = F.relu(self.conv2(x, edge_index)) 
        x = F.dropout(x, p=self.drop_out,training=self.training)
        
        x = global_add_pool(x, batch)
        # x = self.graphnorm(x)
        
        x= F.relu(self.linear(x))
        x=self.linear2(x)
        
        return x


def trainpredictor(predictor_model, train_loader,optimizer):
    
   
    predictor_model.train()
    # crit=torch.nn.MSELoss(reduction='sum')
    loss_fct=torch.nn.SmoothL1Loss(reduction='mean',beta=1)
    total_loss = total_examples = 0
    for data in train_loader:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device) 
        data.y=data.y.to(device)
        optimizer.zero_grad()
        output = predictor_model(data.x, data.edge_index, data.batch)
        loss = loss_fct(output, data.y) 
        if torch.cuda.device_count() > 1 and config["param"]["use_paralell"]=="yes":
             loss.mean().backward()
        else:
            loss.backward() 
    
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        
    print("loss =",loss / len(train_loader.dataset))
    return loss.item()  


@torch.no_grad()
def testpredictor(model,loader,title):
        model.eval()
        ped_list, label_list = [], []
        for data in loader:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            pred = model(data.x, data.edge_index, data.batch)
            ped_list = np.append(ped_list, pred.cpu().detach().numpy())
            label_list = np.append(label_list, data.y.cpu().detach().numpy())
        predictor_R2_Score,predictor_mse,predictor_mae,predictor_corr= evaluate_model_predictor(label_list,ped_list,title)
        
        return  predictor_R2_Score,predictor_mse,predictor_mae,predictor_corr


 
@torch.no_grad()
def predict_accuracy_using_graph(model,graphLoader): 
        
        model.eval()
        prediction_dict={}
        prediction_dict['model_config']=[]
        prediction_dict["Accuracy"]=[]        
        k=int(config["param"]["k"])
        i=0

        for data in graphLoader:
            accuracy=[]
            i+=1
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            pred = model(data.x, data.edge_index, data.batch)
            accuracy= np.append(accuracy, pred.cpu().detach().numpy())
            choices= data.model_config_choices
            # print("Choice is: ",choices)
            choice=[]
            for a in range(len(pred)):  # loop for retriving the GNN configuration of each graph in the data loader
                temp_list=[]
                for key,values in choices.items():
                    # temp_list.append((key,values[1][a].item()))
                    temp_list.append((key, values[a][1]))
                choice.append(temp_list)
            prediction_dict['model_config'].extend(choice)
            prediction_dict["Accuracy"].extend(accuracy)
            # print("choice is",choice)
            # print("acc=",round(float(model_acc[1].item()),2),"Acctype=",type(model_acc[1].item()))

        df = pd.DataFrame.from_dict(prediction_dict)
        TopK=df.nlargest(n=k,columns='Accuracy',keep="all")
        TopK=df[:k]
        # print(TopK)
        return TopK




def get_prediction_from_graph(e_search_space):
   search_metric = config["param"]["search_metric"]
   dim=int(config["predictor"]["dim"])
   drop_out=float(config["predictor"]["drop_out"])
   lr=float(config["predictor"]["lr"]) 
   wd=float(config["predictor"]["wd"])
   num_epoch =int(config["predictor"]["num_epoch"]) 
   best_loss_param_path =f"{config['path']['performance_distribution_folder']}/best_dist_param.pth"
   k=int(config["param"]["k"])
   predict_sample=int(config["param"]["predict_sample"])
   n_sample= int(config["param"]["n"])
   graphlist = []
   bestY=0
   for filename in glob.glob(config['path']['predictor_dataset_folder']+'/*'):
    data=torch.load(filename)
    data.y=data.y.view(-1,1)
    graphlist.append(data)
    if bestY< data.y.item():
        bestY=data.y.item()
        
   print("best Y=",bestY)
   random.shuffle(graphlist)
  
   graph_list=graphlist[:n_sample]
   val_size =int(len(graph_list)*20/100)
   
   val_dataset = graph_list[:val_size]
   train_dataset = graph_list[val_size:]
   
   print("The size of the dataset is",len(graph_list))
   print(f" Predictor training dataset size is :{train_dataset[0].x.shape}")
   print(f'size of predictor training dataset: {len(train_dataset)}')
   print(f'size of predictor validation dataset: {len(val_dataset)}')
   
   start_train_time = time.time()
   
   if config["param"]["encoding_method"] =="embedding":
      
     feature_size=int(config["param"]["nfcode"])+int(config["param"]["noptioncode"])
    
   else:
    
       if config["param"]["feature_size_choice"] =="total_functions":
           feature_size=int(config["param"]["total_function"]) 
           
       if config["param"]["feature_size_choice"] =="total_choices":
           feature_size=int(config["param"]["total_choices"])
          
   model = Predictor(feature_size, dim, 1,drop_out)
   if torch.cuda.device_count() > 1 and config["param"]["use_paralell"]=="yes":
        model = nn.DataParallel(model)
   model.to(device)
   set_seed()
   
   
  
   train_loader= DataLoader(train_dataset, batch_size=Batch_Size,shuffle=True)
   
   val_loader= DataLoader(val_dataset, batch_size=Batch_Size,shuffle=False)
   optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
   
   print("starting training the predictor ...")
   a=0 
   mo=-1

   set_seed() 
   best_loss=999
   for epoch in range(num_epoch):
           loss=trainpredictor(model, train_loader,optimizer)
                             
           if loss <best_loss:
               best_loss = loss
               torch.save(model.state_dict(),best_loss_param_path )
                    
   best_model=Predictor(feature_size, dim, 1,drop_out).to(device)
   best_model.load_state_dict(torch.load(best_loss_param_path))      
   
   R2_Score_tr,pearson_tr,kendall_tr,spearman_tr = testpredictor(best_model,train_loader,title="Predictor training test")
   # good_predictor=model
   add_config("results","R2_Score_train", R2_Score_tr)
   add_config("results","pearson_train", pearson_tr)
   add_config("results","kendall_train", kendall_tr)
   add_config("results","spearman_train", spearman_tr) 
       
  
   # Trained_model = Predictor(feature_size, dim, 1,drop_out).to(device)   
   # Trained_model.load_state_dict(torch.load(f'{config["path"]["predictor_weight_path"]}/PredictorWeights_{config["param"]["run_code"]}.pth'))
   R2_Score_val,pearson_val,kendall_val,spearman_val = testpredictor(best_model,val_loader, title="Predictor validation test")

   
   predictor_training_time = round(time.time() - start_train_time,2)
   add_config("time","predictor_training_time", predictor_training_time)
   add_config("results","R2_Score_val", R2_Score_val)
   add_config("results","pearson_val", pearson_val)
   add_config("results","kendall_val", kendall_val)
   add_config("results","spearman_val", spearman_val)
   
     
     
    # print("start predicting model performance ...")
   
   print('\n starting sampling GNN configurations from search space ...\n')
   sample_list = random_sample(e_search_space=e_search_space,n_sample=predict_sample,predictor=True)
    
   # sample_list = sample_models(predict_sample, e_search_space)
    
   lists=[elt for elt in range(0,len(sample_list),int(config["param"]["batch_sample"]))]
   TopK_models=[]
   start_predict_time = time.time() 
   print("Begin predicting  architecture performance...")
   for i in tqdm(lists):
          a=i+int(config["param"]["batch_sample"])
         
          if a >len(sample_list):
              a=len(sample_list)
       
          sample= sample_list[i:a]
         
     #    transform model configuration into graph data
          graph_list =[]
          for model_config in sample:
              
               x = get_nodes_features(model_config,e_search_space)              
               edge_index=get_edge_index(model_config)
               
               graphdata=Data(x=x,edge_index =edge_index,num_nodes=x.shape[0],model_config_choices = deepcopy(model_config))
               graph_list.append(graphdata)     
              
          set_seed()    
          sample_dataset= DataLoader(graph_list, batch_size=Batch_Size,shuffle=False) 
          # Trained_model = Predictor(feature_size, dim, 1,drop_out).to(device)
          
          # Trained_model.load_state_dict(torch.load(f'{config["path"]["predictor_weight_path"]}/PredictorWeights_{config["param"]["run_code"]}.pth'))
          
          set_seed()                  
          best_model=Predictor(feature_size, dim, 1,drop_out).to(device)
          best_model.load_state_dict(torch.load(best_loss_param_path)) 
          TopK = predict_accuracy_using_graph(best_model,sample_dataset)
          TopK_models.append(TopK)
                     
   TopK_model = pd.concat(TopK_models)  
   TopK_model=TopK_model.nlargest(k,search_metric,keep="all")
   TopK_final=TopK_model[:k]
   
   prediction_time= round(time.time() - start_predict_time,2)
   print("\n End architecture performance prediction. ")         
   add_config("time","pred_time",prediction_time)
       
   return TopK_final


def get_prediction_from_table(performance_record, e_search_space):
   predict_sample=int(config["param"]["predict_sample"])
   type_sample =config["param"]["type_sampling"]
   now =config["param"]["run_code"]
   dataset_name =config["dataset"]["dataset_name"]
   k=int(config["param"]["k"])
   encoding_method = config["param"]["encoding_method"]
   n_sample =int(config["param"]["N"])
   start_time = time.time()
   lb_make = LabelEncoder()
   performance_record = pd.read_csv(performance_record)
   df=performance_record
   search_metric = config["param"]["search_metric"]
   # df =(df-df.mean())/df.std()
   x=df.iloc[:, :-1]  
   y= df[search_metric]
  
   # if config["param"]["encoding_method"] =="embedding":
   #         x =round((x-x.mean())/x.std(),5)
           
   for col in x.columns:
       if col in ['gnnConv1','gnnConv2',"aggregation1","aggregation2",'normalize1','normalize2','activation1','activation2','pooling','criterion',"optimizer"]:  # some function sets do not required do be categorized such as lr, dropout
           x[col]=x[col].astype("category")
           if config["param"]["encoding_method"] =="one_hot":
             x[col]=lb_make.fit_transform(x[col])
    
   X_train,  X_test,  y_train,  y_test = train_test_split(x,  y,  test_size = 0.2, random_state=seed)
   start_train_time = time.time()
   # regr1 = MLPRegressor(random_state=1, max_iter=2000).fit(X_train, np.ravel(y_train))
   regr1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300).fit(X_train, np.ravel(y_train,order='C'))
   
   r2_tr1,pearson_tr1,kendall_tr1,spearman_tr1 = evaluate_model_predictor(y_test, regr1.predict(X_test),"AdaBoostRegressor" )
   df1 = pd.DataFrame(list(zip(y_train,  regr1.predict(X_test))), columns =['real_acc', 'predicted_acc'])
   print(f"regression with AdaboostRegressor gives R2 score={r2_tr1}|r={pearson_tr1}|rho={spearman_tr1}|tau={kendall_tr1}")
  
  
   regr2 = RandomForestRegressor(n_estimators = 300).fit(X_train,  np.ravel(y_train)) 
   r2_tr2,pearson_tr2,kendall_tr2,spearman_tr2 = evaluate_model_predictor(y_test, regr2.predict(X_test),"RandomForestRegressor" )
   df2 = pd.DataFrame(list(zip(y_train,  regr2.predict(X_test))), columns =['real_acc', 'predicted_acc'])
   print(f"regression with RandomForestRegressor gives R2_score={r2_tr2}|r={pearson_tr2}|rho={spearman_tr2}|tau={kendall_tr2}")
   # r2_1,r2_3=0,0
   
   regr3 =  MLPRegressor(random_state=1, max_iter=3000).fit(X_train, np.ravel(y_train))
   r2_tr3,pearson_tr3,kendall_tr3,spearman_tr3 = evaluate_model_predictor(y_test, regr3.predict(X_test),"MLPRegressor" )
   df3 = pd.DataFrame(list(zip(y_train,  regr3.predict(X_test))), columns =['real_acc', 'predicted_acc'])
   print(f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}")
   
   regr4 =  SGDRegressor().fit(X_train, np.ravel(y_train))
   r2_tr4,pearson_tr4,kendall_tr4,spearman_tr4 = evaluate_model_predictor(y_test, regr4.predict(X_test),"LassoCV" )
   df4 = pd.DataFrame(list(zip(y_train,  regr4.predict(X_test))), columns =['real_acc', 'predicted_acc'])
   print(f"regression with SGD gives R2 score={r2_tr4}|r={pearson_tr4}|rho={spearman_tr4}|tau={kendall_tr4}")
   
   
   with open("regression_result1.doc","w") as regr_:
       regr_.write("test set result")
       regr_.write(f"regression with RandomForestRegressor gives R2_score={r2_tr2}|r={pearson_tr2}|rho={spearman_tr2}|tau={kendall_tr2}\n")
       regr_.write(f"regression with AdaboostRegressor gives R2 score={r2_tr1}|r={pearson_tr1}|rho={spearman_tr1}|tau={kendall_tr1}\n")
       regr_.write(f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}\n")
       regr_.write(f"regression with SGD gives R2 score={r2_tr4}|r={pearson_tr4}|rho={spearman_tr4}|tau={kendall_tr4}\n")
   
    
   
   regr1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300).fit(X_train, np.ravel(y_train,order='C'))
   
   r2_tr1,pearson_tr1,kendall_tr1,spearman_tr1 = evaluate_model_predictor(y_train, regr1.predict(X_train),"AdaBoostRegressor" )
   # df1 = pd.DataFrame(list(zip(y_train,  regr1.predict(X_test))), columns =['real_acc', 'predicted_acc'])
   print(f"regression with AdaboostRegressor gives R2 score={r2_tr1}|r={pearson_tr1}|rho={spearman_tr1}|tau={kendall_tr1}")
  
  
  
   regr2 = RandomForestRegressor(n_estimators = 200).fit(X_train,  np.ravel(y_train)) 
   r2_tr2,pearson_tr2,kendall_tr2,spearman_tr2 = evaluate_model_predictor(y_train, regr2.predict(X_train),"RandomForestRegressor" )
   print(f"regression with RandomForestRegressor gives R2_score={r2_tr2}|r={pearson_tr2}|rho={spearman_tr2}|tau={kendall_tr2}")
   
   regr3 =  MLPRegressor(random_state=1, max_iter=2000).fit(X_train, np.ravel(y_train))
   r2_tr3,pearson_tr3,kendall_tr3,spearman_tr3 = evaluate_model_predictor(y_train, regr3.predict(X_train),"MLPRegressor" )
   print(f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}")
   
   regr4 =  LassoCV(cv=8).fit(X_train, np.ravel(y_train))
   r2_tr4,pearson_tr4,kendall_tr4,spearman_tr4 = evaluate_model_predictor(y_train, regr4.predict(X_train),"LassoCV" )
   print(f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}")
   
   
   with open("regression_result2.doc","w") as regr_:
       regr_.write("train set result")
       regr_.write(f"regression with RandomForestRegressor gives R2_score={r2_tr2}|r={pearson_tr2}|rho={spearman_tr2}|tau={kendall_tr2}\n")
       regr_.write(f"regression with AdaboostRegressor gives R2 score={r2_tr1}|r={pearson_tr1}|rho={spearman_tr1}|tau={kendall_tr1}\n")
       regr_.write(f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}\n")
       regr_.write(f"regression with SGD gives R2 score={r2_tr4}|r={pearson_tr4}|rho={spearman_tr4}|tau={kendall_tr4}\n")

   regr=regr2
   R2_Score_tr =r2_tr2
   pearson_tr= pearson_tr2
   kendall_tr =kendall_tr2
   spearman_tr =spearman_tr2
   regressor="RandomForestRegressor"
   df=df1     

   print(f"Selected regressor: {regressor} --R2_Score = {R2_Score_tr}")
 
   predictor_training_time = round(time.time() - start_train_time,2)
   add_config("time","predictor_training_time",predictor_training_time)
   add_config("time","predictor_training_time",predictor_training_time)
   add_config("results","R2_Score_train",R2_Score_tr)
   add_config("results","pearson_train",pearson_tr)
   add_config("results","kendall_train",kendall_tr)
   add_config("results","spearman_train",spearman_tr)  
     
   #----- Predire la performnance de tous les models present dans le search space
   print("start prediction.")
   start_predict_time = time.time()
   sample_list = random_sample(e_search_space=e_search_space,n_sample=predict_sample,predictor=True)
   print(f"Predicting performance for {len(sample_list)} architecture...")
   lists=range(0,len(sample_list),int(config["param"]["batch_sample"]))
   TopK_model=[]
   
  
   for i in tqdm(lists):
         a=i+int(config["param"]["batch_sample"])
         if a >len(sample_list):
             a=len(sample_list)-1
      
         sample= sample_list[i:a]
    
   #    
         predictor_dataset=defaultdict(list)
         for model_config in sample:
            for function,option in model_config.items():
                if config["param"]["encoding_method"] =="one_hot":
                    predictor_dataset[function].append(option[0])
                elif config["param"]["encoding_method"] =="embedding":
                   predictor_dataset[function].append(option[2])
                   
         df =pd.DataFrame.from_dict(predictor_dataset)        
         df = df.sample(frac=1).reset_index(drop=True)  
         df_temp =df.copy(deep=True)
         # if config["param"]["encoding_method"] =="embedding":
         #    df_temp =round((df_temp-df_temp.mean())/df_temp.std(),5)
           
         for col in df_temp.columns:
           if col in ['gnnConv1','gnnConv2',"aggregation1","aggregation2",'normalize1','normalize2','activation1','activation2','pooling','criterion',"optimizer"]:  # some function sets do not required do be categorized such as lr, dropout
               df_temp[col]=df_temp[col].astype("category")
               if config["param"]["encoding_method"] =="one_hot":
                 df_temp[col]=lb_make.fit_transform(df_temp[col])
    
         predicted_accuracy = regr.predict(df_temp)
         df[search_metric]=predicted_accuracy
         TopK=df.nlargest(k,search_metric,keep="all")
         TopK=TopK[:k]
         TopK_model.append(TopK)

   TopK_models = pd.concat(TopK_model)  
  
   TopK_models=TopK_models.nlargest(k,search_metric,keep="all")
   TopK_final=TopK_models[:k].sample(frac=1).reset_index(drop=True) 
   prediction_time= round(time.time() - start_predict_time,2)
   print(TopK_final)  
   add_config("time","pred_time",prediction_time)
   print(" End of prediction")
   return TopK_final

def evaluate_model_predictor(y_train,  y_pred,title="Predictor training"):
   search_metric = config["param"]["search_search_metric"]
   dataset_name =config["dataset"]["dataset_name"]
   n_sample =int(config["param"]["N"])
   # now =config["param"]["now"]
   predictor_eval={}
   
   # R2_Score=round(math.sqrt(mean_squared_error(y_train,  y_pred)),2)
   pearson = round(stats.pearsonr(np.squeeze(y_train), np.squeeze(y_pred))[0],2)
   kendalltau = round(stats.kendalltau(y_train, y_pred)[0],2)     
   spearmanr = round(stats.spearmanr(y_train, y_pred)[0],2)
   r2score = round(r2_score(y_train, y_pred),2)
   
   if title =="Predictor training test":
       col="red"
   elif title=="Predictor validation test":
       col="dodgerblue"
   elif title=="Predictor evaluation test":
       col="limegreen"
   else:
       col="red"
   
   
      # Visualising the Test set results

   # nb=np.array([i for i in range(min1,min1)])
   plt.figure(figsize=(8, 8))

   a=min(y_pred)
   b=min(y_train)
   
   xmin= min(min(y_pred),min(y_train))
   xmax=max(max(y_pred),max(y_train))
   
   if xmin<0:
        xmin=0
   if xmax>100:
         xmax=100
       
   lst =[a for a in range(int(xmin),int(xmax)+1)]
   # lst =[a for a in range(0,100)]
   plt.plot(lst,  lst,  color='black', linewidth=0.6)
   plt.scatter(y_train, y_pred,  color=col, linewidth=0.8)

   plt.title(f'(r={round(pearson,2)},rho={round(spearmanr,2)},tau={kendalltau})',y=1.02,size=28,fontname="Arial Black")#,R2_Score={R2_Score}
   plt.xlabel(f'True {search_metric}',fontsize=28,fontname="Arial Black")
   plt.ylabel(f'Predicted {search_metric}',fontsize=28,fontname="Arial Black")#,fontweight = 'bold'
   # plt.legend()
   plt.grid()
   plt.show()
   plt.savefig(f'{config["path"]["plots_folder"]}/{title}_{dataset_name}.pdf',bbox_inches="tight",dpi=1000)
   return r2score,pearson,kendalltau,spearmanr

