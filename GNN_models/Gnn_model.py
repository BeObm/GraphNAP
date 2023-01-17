import torch
import numpy as np
 
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GENConv, CGConv, GraphConv
from torch_geometric.nn import MessagePassing
from matplotlib.pyplot import plot as plt
from torch.nn import Linear
from sklearn.metrics import accuracy_score#, precision_score, recall_score
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid


gnn_conv=GraphConv
# data=torch.load("data.pt")

dataset = Planetoid(root='data/', name='Cora')
data=dataset[0]
print(data)

if config["param"]["use_GPU"]=="yes":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
   device = torch.device('cpu')



class GNN(MessagePassing):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__(aggr='add')
        
        self.lin_node = Linear(data.num_node_features, hidden_channels)
        self.conv1 = gnn_conv(hidden_channels, hidden_channels)
        self.conv2 = gnn_conv(hidden_channels, dataset.num_classes)
        
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.lin_node(x)
       
        # print(f"x[0] size is: {x[0].size(-1)} | edge_attrib size is {edge_weight.size(-1)}")

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x,0.0, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)


model= GNN(128).to(device)
data=  data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=0.0005),
    dict(params=model.conv2.parameters(), weight_decay=0.0005)
], lr=0.01)  # Only perform weight-decay on first convolution.


criterion = torch.nn.CrossEntropyLoss()

def train(data):
    # optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    model.train()
    out=model(data)
    train_mask=data.train_mask.bool()
    loss=criterion(out[train_mask],data.y[train_mask])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss

def test(typ,data):
    model.eval()
    out=model(data)
    if typ=='val':
        mask=data.val_mask.bool()
    if typ=='test':
        mask=data.test_mask.bool()
    pred=out.argmax(dim=1)
    test_correct= pred[mask]==data.y[mask]
    test_acc=int(test_correct.sum())/int(mask.sum())
    return test_acc*100  

print("starting training ...")
losslist=[]
for epoch in range(1, 101):
   
    loss= train(data)
    vauc= test('val',data)
    tauc= test('test',data)
   
    print(f'Epoch: {epoch:03d}, loss: {loss:.4f}|, | val_Acc= {vauc:.2f} | test_Acc = {tauc:.2f}') #precision: {precision:.4f}| recall: {recall:.4f}')

