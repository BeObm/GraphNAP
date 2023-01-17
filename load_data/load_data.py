import sys

import torch
import torch_geometric
from sklearn.model_selection import StratifiedKFold
from torch import cat
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, PPI, Planetoid,Coauthor,Amazon,Flickr,FacebookPagePage
from settings.config_file import *
import pickle

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")
set_seed()


def get_dataset(type_task, dataset_root, dataset_name,normalize_features=True, transform=None):
    set_seed()
    support_dataset_list ={"node_classification":["Cora", "Citeseer", "Pubmed"],
                           "graph_classification":["DD","PROTEINS","ENZYMES"],
                           "graph_anomaly":["yelp","elliptic"]

                           }
    if dataset_name in support_dataset_list[type_task]:

        if dataset_name in ["Cora", "Citeseer", "Pubmed"]:
            dataset = Planetoid(root=dataset_root, name=dataset_name)#
            if transform is not None and normalize_features:
               dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
            elif normalize_features==True:
                dataset.transform = T.NormalizeFeatures()
            elif transform is not None:
                dataset.transform = transform

        elif dataset_name in ["DD","PROTEINS","ENZYMES"]:
            dataset = TUDataset(root=dataset_root, name=dataset_name) #,use_node_attr=True,use_edge_attr=True
        elif dataset_name == 'molecule':
            dataset=MoleculeDataset(dataset_root, dataset_name)
            dataset.num_classes = dataset[0].num_classes

        elif dataset_name in ["yelp","elliptic"]:
            dataset = pickle.load(open(f'{dataset_root}/{dataset_name}.dat', 'rb'))
            dataset.num_classes=2
    else:
        print(f"@@@@@@@@@@  The {dataset_name} dataset is not supported for {type_task} by the current version ")
        sys.exit()

    return dataset
    

def load_dataset(dataset, batch_dim=Batch_Size):
    """
    Parameters
    ----------
    dataset : TYPE tuple
        DESCRIPTION. (root, filename)
    split : TYPE, optional  list
        DESCRIPTION. The default is [0.8,0.1,0.1].
    batch_size : TYPE, optional
        DESCRIPTION. The default is 64.

    Returns
    -------
    dataset : TYPE
        DESCRIPTION.
    train_loader : TYPE
        DESCRIPTION.
    val_loader : TYPE
        DESCRIPTION.
    test_loader : TYPE
        DESCRIPTION.

    """
    type_task =config["dataset"]["type_task"]
    set_seed()


    if type_task == "node_classification":
        if config["param"]["learning_type"]=="supervised":

               test_loader=train_loader=val_loader= Load_nc_data(dataset[0])
        else:
            test_loader=train_loader=val_loader= dataset[0]
    elif type_task =="graph_anomaly":
        # test_loader = train_loader = val_loader = dataset
        test_loader = train_loader = val_loader = Load_nc_data(dataset)
    else: 
        n = int(len(dataset)*20/100)
        test_dataset = dataset[:n]
        val_dataset = dataset[n:2*n]
        train_dataset = dataset[2*n:]
        train_loader = DataLoader(train_dataset, batch_size=batch_dim,shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_dim,shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_dim,shuffle=False)
        add_config("dataset", "len_traindata", len(train_dataset))
        add_config("dataset", "len_testdata", len(test_dataset))
        add_config("dataset", "len_valdata", len(val_loader))

    return train_loader, val_loader, test_loader



def Load_nc_data(data,shuffle=True):
     
     if shuffle==True:
        set_seed()

        indices = torch.randperm(data.x.size(0))
        data.train_mask = index_to_mask(indices[1500:3500], size=data.num_nodes)
        data.val_mask = index_to_mask(indices[1000:1500], size=data.num_nodes)
        data.test_mask = index_to_mask(indices[:1000], size=data.num_nodes)
     else:
        # data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        # data.train_mask[:1000] = 1
        # data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        # data.val_mask[1000: 1500] = 1
        # data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        # data.test_mask[1500:2000] = 1
        set_seed()

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.train_mask[:-1000] = 1
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.val_mask[-1000: -500] = 1
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.test_mask[-500:] = 1
     return data








def Load_nc_data2(data):

    set_seed()

    data=data[0]
    skf = StratifiedKFold(10, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]

    split = [cat(idx[:6], 0), cat(idx[6:8], 0), cat(idx[8:], 0)]
    
    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)
    return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.uint8, device=index.device)
    mask[index] = 1
    return mask