# -*- coding: utf-8 -*-

from settings.config_file import *

import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.norm import GraphNorm, InstanceNorm,BatchNorm
from torch_geometric.nn import GENConv, GraphUNet,HypergraphConv,GraphConv,CGConv,GATConv,GCNConv,TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool,global_add_pool,GlobalAttention
from torch_geometric.nn import SuperGATConv,GCN2Conv,GINEConv,PNAConv,DeepGCNLayer,FiLMConv
from torch_geometric.nn import GatedGraphConv,SAGEConv,ChebConv,ResGatedGraphConv
from torch_geometric.nn import FastRGCNConv,RGCNConv,MFConv,APPNP,SGConv,ARMAConv,TAGConv,AGNNConv,GATv2Conv
from torch_geometric.nn import FeaStConv,PPFConv,XConv,DynamicEdgeConv,EdgeConv,NNConv,SplineConv,GMMConv,PointNetConv
from torch_geometric.nn import HGTConv,GeneralConv,PDNConv,EGConv,FAConv,WLConv,PANConv,ClusterGCNConv,LEConv
from torch_geometric.nn import GraphNorm,DenseSAGEConv,DenseGraphConv,DenseGCNConv,MetaLayer
set_seed()

def map_activation_function(function_name):
    if function_name =='ReLU6':
        return nn.ReLU6()
    elif function_name == "linear":
        return lambda x: x
    elif function_name == "Elu":
        return nn.ELU()
    elif function_name == "Sigmoid":
        return nn.Sigmoid()
    elif function_name == "Tanh":
        return nn.Tanh()
    elif function_name =="leaky_relu":
           return nn.LeakyReLU()
    elif function_name =="PReLU":
           return nn.PReLU()  
    elif function_name =="Softplus":
           return nn.Softplus() 
    elif function_name =="Relu":
           return nn.ReLU() 
    else:
        raise Exception(f"{function_name} is a wrong activate function")



def map_pooling(pooling):
    if pooling =="global_mean_pool":
        return global_mean_pool
    if pooling =="global_max_pool":
       return global_max_pool
    if pooling =="GlobalAttention":
       return GlobalAttention
    if pooling =="global_add_pool":
        return global_add_pool
 
        
 
def map_normalization(normalizer):
    if normalizer=="GraphNorm":
        return GraphNorm
    elif normalizer=="InstanceNorm":
       return InstanceNorm
    elif normalizer=="BatchNorm":
        return BatchNorm
    elif normalizer=="False" or normalizer==False:
        return False
    
 
def map_gnn_model(gnn_model):
    
    if gnn_model == 'GCNConv':
        return (GCNConv,gnn_model)
  
    elif gnn_model == 'GENConv':
        return (GENConv,gnn_model)
   
    elif gnn_model == 'GraphConv':
        return (GraphConv,gnn_model)
   
    elif gnn_model == 'CGConv':
        return (CGConv,gnn_model)
    
    elif gnn_model == 'SGConv':
        return (SGConv,gnn_model)
    
    elif gnn_model == 'GATConv':
        return (GATConv,gnn_model)
    
    elif gnn_model == 'TAGConv':
         return (TAGConv,gnn_model)
     
    elif gnn_model == 'ClusterGCNConv':
            return (ClusterGCNConv,gnn_model)
        
    elif gnn_model == 'LEConv':
         return (LEConv,gnn_model)
     
    elif gnn_model =="linear":
            return (LinearConv,gnn_model)
    elif gnn_model =="GINEConv":
            return (GINEConv,gnn_model)
    elif gnn_model == 'gat_sym' or gnn_model == 'ChebConv' :
         return (ChebConv,gnn_model)   
    elif gnn_model == 'SplineConv':
         return (SplineConv,gnn_model)
     
    elif gnn_model == 'AGNNConv':
         return (AGNNConv,gnn_model)
    elif gnn_model == 'HypergraphConv':
        return (HypergraphConv,gnn_model)
    
    elif gnn_model == 'TransformerConv':
         return (TransformerConv,gnn_model)
   
    elif gnn_model == 'SuperGATConv': 
         return (SuperGATConv,gnn_model)
   
    elif gnn_model == 'GCN2Conv':
         return (GCN2Conv,gnn_model)
    elif gnn_model == 'GINEConv':
        return (GINEConv,gnn_model)
    
    elif gnn_model == 'PNAConv':
        return (PNAConv,gnn_model)
     
    elif gnn_model == 'FiLMConv':
        return (FiLMConv,gnn_model)

    elif gnn_model == 'DeepGCNLayer':
        return (DeepGCNLayer,gnn_model)
     
    elif gnn_model == 'SAGEConv':
        return (SAGEConv,gnn_model)
   
    elif gnn_model == 'GraphConv':
        return (GraphConv,gnn_model)
   
    
    elif gnn_model == 'GatedGraphConv':
        return (GatedGraphConv,gnn_model)
    
    elif gnn_model == 'ResGatedGraphConv':
        return (ResGatedGraphConv,gnn_model)

    elif gnn_model == 'GATv2Conv':
        return (GATv2Conv,gnn_model)
     
    
   
    elif gnn_model == 'FastRGCNConv':
        return (FastRGCNConv,gnn_model)
   
    elif gnn_model == 'RGCNConv':
        return (RGCNConv,gnn_model)
    
    elif gnn_model == 'MFConv':
        return (MFConv,gnn_model)
    
    elif gnn_model == 'PPFConv':
        return (PPFConv,gnn_model)
    
    elif gnn_model == 'XConv':
        return (XConv,gnn_model)
    
    elif gnn_model == 'DynamicEdgeConv':
        return (DynamicEdgeConv,gnn_model)

    elif gnn_model == 'EdgeConv':
        return (EdgeConv,gnn_model)
     
    elif gnn_model == 'CGConv':
        return (CGConv,gnn_model)
   
    elif gnn_model == 'NNConv':
         return (NNConv,gnn_model)
   
    elif gnn_model == 'GMMConv':
        return (GMMConv,gnn_model)
    
    elif gnn_model == 'PointNetConv':
        return (PointNetConv,gnn_model)
    elif gnn_model == 'FeaStConv':
        return (FeaStConv ,gnn_model)

    elif gnn_model == 'HypergraphConv':
        return (HypergraphConv,gnn_model)
    elif gnn_model == 'GCN2Conv':
        return (GCN2Conv,gnn_model)
    
    elif gnn_model == 'WLConv':
        return (WLConv,gnn_model)
    
    elif gnn_model == 'FAConv':
        return (PointNetConv,gnn_model)
    elif gnn_model == 'FAConv':
        return (FeaStConv ,gnn_model)

    elif gnn_model == 'EGConv':
        return (EGConv,gnn_model)
    
    elif gnn_model == 'PDNConv':
        return (PDNConv ,gnn_model)  
    
    elif gnn_model == 'GeneralConv':
        return (GeneralConv,gnn_model)
    elif gnn_model == 'HGTConv':
        return (HGTConv,gnn_model) 
     
 
    elif gnn_model == 'DenseGCNConv':
        return (DenseGCNConv,gnn_model)
   
    elif gnn_model == 'DenseGraphConv':
        return (DenseGraphConv,gnn_model)
    
    elif gnn_model == 'DenseSAGEConv':
        return (DenseSAGEConv,gnn_model)
    elif gnn_model == 'FAConv':
        return (FeaStConv ,gnn_model)

  
     
    elif gnn_model == 'ARMAConv':
        return (ARMAConv,gnn_model)
   
    elif gnn_model == 'PANConv':
        return (PANConv,gnn_model)
    elif gnn_model == 'GraphUNet':
        return (GraphUNet ,gnn_model)

    elif gnn_model == 'GraphNorm':
        return (GraphNorm,gnn_model)
   
    elif gnn_model == 'FiLMConv':
        return (FiLMConv,gnn_model)    
 
    else:
        raise ValueError(f"{gnn_model} is a wrong convolution type name")
 

class LinearConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


   
def map_criterion(criterion):
    if criterion=='CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif criterion =="CTCLoss":  
        return torch.nn.CTCLoss()
    elif criterion =="MultiMarginLoss":  
        return torch.nn.MultiMarginLoss()
    elif criterion =="BCEWithLogitsLoss":  
        return torch.nn.BCEWithLogitsLoss()
    elif criterion =="fn_loss":
        return F.nll_loss
    else:
        map_function_error(criterion)
        
        
def map_optimizer(optimizers,model,lr,weight_decay=False):
    if optimizers =='adam':
        return torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    elif optimizers =='sgd':
        return torch.optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay)
    else:
        map_function_error(optimizers)
        
        
        
# def map_train(model):
#     if model =='GCN':
#         return 'train_GCN'
#     elif model=='GCN2':
#         return "tain_GCN2"
        
def map_function_error(function):
    raise Exception("Error: {} is a wrong value".format(function))
 
