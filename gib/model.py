# https://github.com/Samyu0304/graph-information-bottleneck-for-Subgraph-Recognition/tree/main

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_adj


class GIBGIN(nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GIBGIN, self).__init__()
        
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden)), 
            train_eps = False)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden)), 
                    train_eps=False))
        
        # subgraph generator layer (generate assignment matrix)
        self.cluster1 = Linear(hidden, hidden)
        self.cluster2 = Linear(hidden, 2)
        
        # classifier
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.mse_loss = nn.MSELoss()
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.cluster1.reset_parameters()
        self.cluster2.reset_parameters()
    
    def assignment(self, x):
        return self.cluster2(torch.tanh(self.cluster1(x)))
    
    def aggregate(self, assignment, x, batch, edge_index):
        
        max_id = torch.max(batch)
        EYE = torch.ones(2).to(edge_index.device)
        
        all_adj = to_dense_adj(edge_index, max_num_nodes = len(batch))[0]

        all_con_penalty = 0
        all_sub_embedding = []
        all_graph_embedding = []

        st = 0
        end = 0

        for i in range(int(max_id + 1)):
            j = 0
            while batch[st + j] == i and st + j <= len(batch) - 2:
                j += 1
            end = st + j

            if end == len(batch) - 1: 
                end += 1

            one_batch_x = x[st:end]
            one_batch_assignment = assignment[st:end]

            subgraph_features = torch.mm(torch.t(one_batch_assignment), one_batch_x)
            subgraph_features = subgraph_features[0].unsqueeze(dim = 0) # S^T X: represetation of g_sub

            Adj = all_adj[st:end, st:end]
            new_adj = torch.mm(torch.t(one_batch_assignment), Adj)
            new_adj = torch.mm(new_adj, one_batch_assignment)
            normalize_new_adj = F.normalize(new_adj, p = 1, dim = 1)
            norm_diag = torch.diag(normalize_new_adj)
            mse_loss = nn.MSELoss()
            con_penalty = mse_loss(norm_diag, EYE) # connectivity loss

            graph_embedding = torch.mean(one_batch_x, dim = 0, keepdim = True)
            # graph_embedding = torch.mean(x, dim = 0, keepdim = True)

            all_sub_embedding.append(subgraph_features)
            all_graph_embedding.append(graph_embedding)

            all_con_penalty = all_con_penalty + con_penalty

            st = end

        all_sub_embedding = torch.cat(tuple(all_sub_embedding), dim = 0)
        all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim = 0)
        all_con_penalty = all_con_penalty / (max_id + 1)
        
        return all_sub_embedding, all_graph_embedding, all_con_penalty
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        h = self.conv1(x, edge_index)
        for conv in self.convs:
            h = conv(h, edge_index)
        
        assignment = F.softmax(self.assignment(h), dim = 1)
        
        all_sub_embedding, all_graph_embedding, all_con_penalty = self.aggregate(assignment, h, batch, edge_index)
        
        h = F.relu(self.lin1(all_sub_embedding))
        h = F.dropout(h, p = 0.5, training = self.training)
        h = self.lin2(h)
        out = F.log_softmax(h, dim = -1)
        
        return out, all_sub_embedding, all_graph_embedding, all_con_penalty
    
    def __repr__(self):
        return self.__class__.__name__


# for optimizing the graph and subgraph (phi_2)
class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        
        self.input_size = 2 * hidden_size
        self.hidden_size = hidden_size
        
        self.lin1 = Linear(self.input_size, self.hidden_size)
        self.lin2 = Linear(self.hidden_size, 1)
        self.relu = ReLU()
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        
    def forward(self, graph_embeddings, subgraph_embeddings):
        cat_embeddings = torch.cat((graph_embeddings, subgraph_embeddings), dim = -1)
        
        pre = self.relu(self.lin1(cat_embeddings))
        pre = self.relu(self.lin2(pre))
        
        return pre
