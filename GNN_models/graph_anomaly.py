# -*- coding: utf-8 -*-


from torch.nn import Linear
# from torch_geometric.nn.models import MLP
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import inspect
from settings.config_file import *

set_seed()
from search_algo.utils import aucPerformance


class GNN_Model(MessagePassing):
    def __init__(self, param_dict):
        super(GNN_Model, self).__init__()
        self.dataset = param_dict['dataset']
        self.aggr1 = param_dict['aggregation1']
        self.aggr2 = param_dict['aggregation2']
        self.hidden_channels1 = int(param_dict['hidden_channels1'])
        self.hidden_channels2 = int(param_dict['hidden_channels2'])
        self.head1 = param_dict['multi_head1']
        self.head2 = param_dict['multi_head2']
        self.type_task = param_dict['type_task']
        self.gnnConv1 = param_dict['gnnConv1'][0]
        self.gnnConv2 = param_dict['gnnConv2'][0]

        self.edge_attr_size = self.dataset.num_edge_features
        if self.edge_attr_size == 0:
            self.edge_attr_size = 1

        self.activation1 = param_dict['activation1']
        self.activation2 = param_dict['activation2']
        self.dropout1 = param_dict['dropout1']
        self.dropout2 = param_dict['dropout2']
        self.normalize1 = param_dict['normalize1']
        self.normalize2 = param_dict['normalize2']

        if param_dict['gnnConv1'][1] == "linear":
            self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1)
            self.input_conv2 = self.hidden_channels1
        else:
            if 'head' in inspect.getfullargspec(self.gnnConv1)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv1)[0]:
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1, head=self.head1,
                                               aggr=self.aggr1, dropout=self.dropout1)
                else:
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1, head=self.head1,
                                               aggr=self.aggr1)
                self.input_conv2 = self.hidden_channels1 * self.head1

            elif 'heads' in inspect.getfullargspec(self.gnnConv1)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv1)[0]:
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1, heads=self.head1,
                                               aggr=self.aggr1, dropout=self.dropout1)
                else:
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1, heads=self.head1,
                                               aggr=self.aggr1)

                self.input_conv2 = self.hidden_channels1 * self.head1

            elif 'num_heads' in inspect.getfullargspec(self.gnnConv1)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv1)[0]:
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1,
                                               num_heads=self.head1, aggr=self.aggr1, dropout=self.dropout1)
                else:
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1,
                                               num_heads=self.head1, aggr=self.aggr1)

                self.input_conv2 = self.hidden_channels1 * self.head1

            else:
                if param_dict['gnnConv1'][1] == "SGConv":  # splineconv does not support multihead
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1, K=2,
                                               aggr=self.aggr1)
                elif param_dict['gnnConv1'][1] == 'ChebConv' or param_dict['gnnConv1'][1] == 'gat_sym':
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1, K=2,
                                               aggr=self.aggr1)
                elif param_dict['gnnConv1'][1] == 'GraphUNet':
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, 32, self.hidden_channels1, depth=3)
                elif param_dict['gnnConv1'][1] == 'RGCNConv':
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1,
                                               num_relations=self.edge_attr_size, aggr=self.aggr1)
                elif param_dict['gnnConv1'][1] == "PDNConv":  # splineconv does not support multihead
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1,
                                               edge_dim=self.edge_attr_size, hidden_channels=self.hidden_channels1)
                elif param_dict['gnnConv1'][1] == "FastRGCNConv":  # splineconv does not support multihead
                    self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1, num_relations=1)

                else:
                    if 'dropout' in inspect.getfullargspec(self.gnnConv1)[0]:
                        self.conv1 = self.gnnConv1(self.dataset.num_node_features, self.hidden_channels1,
                                                   aggr=self.aggr1, dropout=self.dropout1)
                    else:
                        self.conv1 = self.gnnConv1(self.dataset.num_node_features,
                                                   self.hidden_channels1)  # ,aggr=self.aggr1)

                self.input_conv2 = self.hidden_channels1

        # multiple attention head is used and node and edge features have the same dimensiion
        if param_dict['gnnConv2'][1] == "linear":
            self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2)
            self.output_conv2 = self.hidden_channels2
        else:

            if 'head' in inspect.getfullargspec(self.gnnConv2)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv2)[0]:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, head=self.head2,
                                               aggr=self.aggr2, dropout=self.dropout2)
                else:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, head=self.head2,
                                               aggr=self.aggr2)

                self.output_conv2 = self.hidden_channels2 * self.head2
                # if self.edge_attr_size >0:
                #     self.embed_edges = Linear(self.edge_attr_size, self.hidden_channels*self.head2)
            elif 'heads' in inspect.getfullargspec(self.gnnConv2)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv2)[0]:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, heads=self.head2,
                                               aggr=self.aggr2, dropout=self.dropout2)
                else:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, heads=self.head2,
                                               aggr=self.aggr2)
                self.output_conv2 = self.hidden_channels2 * self.head2
                # if self.edge_attr_size >0:
                #     self.embed_edges = Linear(self.edge_attr_size, self.hidden_channels*self.head2)

            elif 'num_heads' in inspect.getfullargspec(self.gnnConv2)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv2)[0]:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, num_heads=self.head2,
                                               aggr=self.aggr2, dropout=self.dropout2)
                else:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, num_heads=self.head2,
                                               aggr=self.aggr2)
                self.output_conv2 = self.hidden_channels2 * self.head2
                # if self.edge_attr_size >0:
                #     self.embed_edges = Linear(self.edge_attr_size, self.hidden_channels*self.head2)
            else:
                if param_dict['gnnConv2'][1] == "SGConv":  # splineconv does not support multihead
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, K=2, aggr=self.aggr2)
                elif param_dict['gnnConv2'][1] == 'ChebConv' or param_dict['gnnConv2'][1] == 'gat_sym':
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, K=2, aggr=self.aggr2)
                elif param_dict['gnnConv2'][1] == 'GraphUNet':
                    self.conv2 = self.gnnConv2(self.input_conv2, 32, self.hidden_channels2, depth=3)
                elif param_dict['gnnConv2'][1] == 'RGCNConv':
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2,
                                               num_relations=self.edge_attr_size, aggr=self.aggr2)
                elif param_dict['gnnConv2'][1] == 'PDNConv':
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, edge_dim=self.edge_attr_size,
                                               hidden_channels=self.hidden_channels2)
                elif param_dict['gnnConv2'][1] == 'FastRGCNConv':
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, num_relations=1)
                else:
                    if 'dropout' in inspect.getfullargspec(self.gnnConv2)[0]:
                        self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, aggr=self.aggr2,
                                                   dropout=self.dropout2)
                    else:
                        self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels2, aggr=self.aggr2)

                self.output_conv2 = self.hidden_channels2

        if self.normalize1 != False:
            self.batchnorm1 = self.normalize1(self.input_conv2)

        if self.normalize2 != False:
            self.batchnorm2 = self.normalize2(self.output_conv2)

        self.linear = Linear(self.output_conv2, self.dataset.num_classes)

    def get_forward_conv(self, num_layer, conv, x, edge_index, edge_attr):
        # meme si le dataset possede le edge attribute, il faut verifier si la convolution l utilise sinon cela ne sera pas utiliser pour une telle convolution

        if ('edge_attr' in inspect.getfullargspec(conv.forward)[0]) or (
                "hypergraph_atrr" in inspect.getfullargspec(conv.forward)[0]) or (
                "edge_attr" in inspect.getfullargspec(conv.forward)[0]):

            if num_layer == 1:
                return self.conv1(x, edge_index, edge_attr)
            if num_layer == 2:
                return self.conv2(x, edge_index, edge_attr)

        else:
            if num_layer == 1:
                return self.conv1(x, edge_index)
            if num_layer == 2:
                return self.conv2(x, edge_index)

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if 'batch' in data.keys:
            batch = data.batch

        x = F.dropout(x, self.dropout1, training=self.training)

        x = self.get_forward_conv(1, self.gnnConv1, x, edge_index, edge_attr)
        if self.normalize1 not in [False, "False"]:
            x = self.batchnorm1(x)
        x = self.activation1(x)
        x = F.dropout(x, self.dropout2, training=self.training)
        x = self.get_forward_conv(2, self.gnnConv2, x, edge_index, edge_attr)

        if self.normalize2 not in [False, "False"]:
            x = self.batchnorm2(x)
        x = self.activation2(x)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


def train_function(model, data, criterion, optimizer):
    model.train()
    data = data.to(device)
    train_mask = data.train_mask.bool()
    optimizer.zero_grad()
    out = model(data)

    train_loss = criterion(out[train_mask], data.y[train_mask])

    if torch.cuda.device_count() > 1 and config["param"]["use_paralell"] == "yes":
        train_loss.backward()
    else:
        train_loss.backward()

    optimizer.step()

    return float(train_loss)  # ,train_acc


@torch.no_grad()
def test_function(model, data, typ="val"):
    model.eval()
    data = data.to(device)
    out = model(data)
    if typ == 'val':
        mask = data.val_mask.bool()
    if typ == 'test':
        mask = data.test_mask.bool()
    pred = out.argmax(dim=1)
    auc_roc, auc_pr = aucPerformance(data.y[mask], pred[mask])

    acc_score_ = acc_score(data.y[mask], pred[mask])
    balanced_acc_score_ = balanced_acc_score(data.y[mask], pred[mask])
    matthews_corrcoef = mcc_score(data.y[mask],pred[mask])

    performance_scores["accuracy_score"] = acc_score_
    performance_scores["balanced_accuracy_score"] = balanced_acc_score_
    performance_scores["matthews_corr_coef"] = matthews_corrcoef

    # test_correct= pred[mask]==data.y[mask]
    # test_acc=int(test_correct.sum())/int(mask.sum())
    return auc_roc, auc_pr


