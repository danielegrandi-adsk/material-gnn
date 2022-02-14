import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax, degree
import random

"""Reproducibility"""
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class GIN(MessagePassing):
    def __init__(self, emb_dim):
        super(GIN, self).__init__()
        self.mlp = torch.nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_attr):  # Here
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)  # MLP


class GCN(MessagePassing):
    def __init__(self, emb_dim):
        super(GCN, self).__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros((x.size(0), edge_attr.size(1)), dtype=edge_attr.dtype).to(x.device)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GAT(MessagePassing):
    def __init__(self, emb_dim):
        super(GAT, self).__init__()
        self.att = nn.Parameter(torch.Tensor(1, 2 * emb_dim))
        glorot(self.att)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros((x.size(0), edge_attr.size(1)), dtype=edge_attr.dtype).to(x.device)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_j = x_j + edge_attr
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index[0])
        return x_j * alpha.view(-1, 1)


class GraphSAGE(MessagePassing):
    def __init__(self, emb_dim):
        super(GraphSAGE, self).__init__(aggr='mean')
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GraphNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, hier_dim, num_class, num_layers, network):
        super(GraphNN, self).__init__()
        self.drop = nn.Dropout(p=0.1)
        self.network = network
        self.hier_dim = hier_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.linear_node = nn.Linear(node_dim, hid_dim)  # Weight*h + bias
        self.linear_edge = nn.Linear(edge_dim, hid_dim)  # Weight*h + bias
        self.output_layer = nn.Sequential(
            nn.Linear(hier_dim + hid_dim, 2 * hid_dim),
            nn.BatchNorm1d(2 * hid_dim),
            nn.PReLU(),
            nn.Linear(2 * hid_dim, num_class)
        )

        self.bn = nn.BatchNorm1d(node_dim)  # Batch normalization for concatenated node features

        self.layers = nn.ModuleList()
        self.lin_layers = nn.ModuleList()

        for layer in range(self.num_layers):
            if self.network == 'gin':
                self.layers.append(GIN(hid_dim))
            elif self.network == 'gcn':
                self.layers.append(GCN(hid_dim))
            elif self.network == 'gat':
                self.layers.append(GAT(hid_dim))
            elif self.network == 'sage':
                self.layers.append(GraphSAGE(hid_dim))
            self.lin_layers.append(nn.Linear(hid_dim, hid_dim))

    def forward(self, x, edge_index, e, c):

        # GraphNN forward() parameters:
        # x = x - node features
        # edge = edge_index - 2 nodes to each edge (compressed version of the adj matrix)
        # e = e - edge features
        # c = 0/y1/y2 - ground truth input predictions (Teacher forcing)

        # different feature space, project to similar feature dimension

        # x = self.bn(x)  # Batch normalization for concatenated node features

        h = self.linear_node(x)  # linear transformation (project from node dimension to hidden dimension)
        # h = self.drop(h)
        e = self.linear_edge(e)  # linear transformation

        h_list = [h]

        for layer in range(self.num_layers):  # for each GNN layer

            if self.network != 'gin':  # default is GIN, so this part can be ignored
                h = self.lin_layers[layer](h_list[layer])

            h = self.layers[layer](h_list[layer], edge_index, e)  # Passing data to one layer of GNN
            # h = self.drop(F.leaky_relu(h, negative_slope=0.2))  # Take node embeddings, activation, drop-out
            h = F.leaky_relu(h, negative_slope=0.2)
            h_list.append(h)

        # take node embeddings of all layers and take sum (residual network - uses skip connections)
        h_list = [h.unsqueeze_(0) for h in h_list]
        h = torch.sum(torch.cat(h_list), 0)

        # if commented out previous two lines, will give the embeddings of the last layer

        if self.hier_dim > 0:
            h = self.output_layer(torch.cat((h, c), dim=-1))
        else:
            h = self.output_layer(h)

        return h


class HierarchicalGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, num_class_l1, num_class_l2, num_class_l3, num_layers, network):
        super(HierarchicalGNN, self).__init__()
        self.gnn_level1 = GraphNN(node_dim, edge_dim, hid_dim, 0, num_class_l1, num_layers, network)
        self.gnn_level2 = GraphNN(node_dim, edge_dim, hid_dim, num_class_l1, num_class_l2, num_layers, network)
        self.gnn_level3 = GraphNN(node_dim, edge_dim, hid_dim, num_class_l2, num_class_l3, num_layers, network)

    def forward(self, x, edge_index, e, y1, y2):
        # gnn.forward() parameters:
        # x = batch.x.float() - node attributes
        # edge_index = batch.edge_index - 2 nodes to each edge
        # e = batch.e.float() - edge attributes
        # y1 = F.one_hot(batch.y1, self.num_class_l1) - tier 1 function
        # y2 = F.one_hot(batch.y2, self.num_class_l2) - tier 2 function

        # GraphNN forward() parameters:
        # x = x - node attributes
        # edge = edge_index - 2 nodes to each edge
        # e = e - edge attributes
        # c = 0/y1/y2 - ground truth input predictions (Teacher forcing)

        # compute tier predictions (softmax?)

        # First MLP no predecessor so just vector of 0 as input predictions

        yp_l1 = self.gnn_level1(x, edge_index, e, 0)
        yp_l2 = self.gnn_level2(x, edge_index, e, y1)
        yp_l3 = self.gnn_level3(x, edge_index, e, y2)
        return yp_l1, yp_l2, yp_l3  # returns tier predictions

    @torch.no_grad()
    def predict(self, x, edge_index, e):
        yp_l1 = F.softmax(self.gnn_level1(x, edge_index, e, 0), dim=-1)
        yp_l2 = F.softmax(self.gnn_level2(x, edge_index, e, yp_l1), dim=-1)
        yp_l3 = F.softmax(self.gnn_level3(x, edge_index, e, yp_l2), dim=-1)
        return yp_l1, yp_l2, yp_l3


class CustomizedGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, num_materials, num_layers, network):
        super(CustomizedGNN, self).__init__()
        self.gnn_level1 = GraphNN(node_dim, edge_dim, hid_dim, 0, num_materials, num_layers, network)

    def forward(self, x, edge_index, e):
        material_predictions = self.gnn_level1(x, edge_index, e, 0)  # 0 here emulate input predictions
        return material_predictions

    @torch.no_grad()
    def predict(self, x, edge_index, e):
        material_predictions = F.softmax(self.gnn_level1(x, edge_index, e, 0), dim=-1)
        return material_predictions


class MLP(nn.Module):
    def __init__(self, node_dim, hid_dim, num_class_l1, num_class_l2, num_class_l3):
        super(MLP, self).__init__()
        self.mlp_level1 = nn.Sequential(
            nn.Linear(node_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hid_dim, num_class_l1)
        )
        self.mlp_level2 = nn.Sequential(
            nn.Linear(node_dim + num_class_l1, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hid_dim, num_class_l2)
        )
        self.mlp_level3 = nn.Sequential(
            nn.Linear(node_dim + num_class_l2, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hid_dim, num_class_l3)
        )

    def forward(self, x, y1, y2):
        yp_l1 = self.mlp_level1(x)
        yp_l2 = self.mlp_level2(torch.cat((x, y1), dim=-1))
        yp_l3 = self.mlp_level3(torch.cat((x, y2), dim=-1))
        return yp_l1, yp_l2, yp_l3

    @torch.no_grad()
    def predict(self, x):
        yp_l1 = F.softmax(self.mlp_level1(x), dim=-1)
        yp_l2 = F.softmax(self.mlp_level2(torchcat((x, yp_l1), dim=-1)), dim=-1)
        yp_l3 = F.softmax(self.mlp_level3(torch.cat((x, yp_l2), dim=-1)), dim=-1)
        return yp_l1, yp_l2, yp_l3


class Linear(nn.Module):
    def __init__(self, node_dim, hid_dim, num_class_l1, num_class_l2, num_class_l3):
        super(Linear, self).__init__()
        self.linear_l1 = nn.Linear(node_dim, num_class_l1)
        self.linear_l2 = nn.Linear(node_dim + num_class_l1, num_class_l2)
        self.linear_l3 = nn.Linear(node_dim + num_class_l2, num_class_l3)

    def forward(self, x, y1, y2):
        yp_l1 = self.linear_l1(x)
        yp_l2 = self.linear_l2(torch.cat((x, y1), dim=-1))
        yp_l3 = self.linear_l3(torch.cat((x, y2), dim=-1))
        return yp_l1, yp_l2, yp_l3

    @torch.no_grad()
    def predict(self, x):
        yp_l1 = F.softmax(self.linear_l1(x), dim=-1)
        yp_l2 = F.softmax(self.linear_l2(torch.cat((x, yp_l1), dim=-1)), dim=-1)
        yp_l3 = F.softmax(self.linear_l3(torch.cat((x, yp_l2), dim=-1)), dim=-1)
        return yp_l1, yp_l2, yp_l3


if __name__ == "__main__":
    pass
