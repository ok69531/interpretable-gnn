import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


class VariationalGIB(nn.Module):
    def __init__(self, args, number_of_features):
        super(VariationalGIB, self).__init__()
        
        self.args = args
        self.subgraph_const = self.args.subgraph_const
        self.number_of_features = number_of_features

        self.mseloss = nn.MSELoss()
        self.relu = nn.ReLU()

        self.graph_convolution_1 = GINConv(
            nn.Sequential(
                nn.Linear(self.number_of_features, self.args.first_gcn_dimensions),
                nn.ReLU(),
                nn.Linear(self.args.first_gcn_dimensions, self.args.first_gcn_dimensions),
                nn.ReLU(),
                nn.BatchNorm1d(self.args.first_gcn_dimensions),
            ), train_eps=False)

        self.graph_convolution_2 = GINConv(
            nn.Sequential(
                nn.Linear(self.args.first_gcn_dimensions, self.args.second_gcn_dimensions),
                nn.ReLU(),
                nn.Linear(self.args.second_gcn_dimensions, self.args.second_gcn_dimensions),
                nn.ReLU(),
                nn.BatchNorm1d(self.args.second_gcn_dimensions),
            ), train_eps=False)

        self.fully_connected_1 = nn.Linear(self.args.second_gcn_dimensions, self.args.first_dense_neurons)
        self.fully_connected_2 = nn.Linear(self.args.first_dense_neurons, self.args.second_dense_neurons)
    
    def reset_parameters(self):
        self.graph_convolution_1.reset_parameters()
        self.graph_convolution_2.reset_parameters()
        self.fully_connected_1.reset_parameters()
        self.fully_connected_2.reset_parameters()

    def gumbel_softmax(self, prob):
        return F.gumbel_softmax(prob, tau = 1, dim = -1)

    def forward(self, data):
        epsilon = 0.0000001

        features = data.x
        edges = data.edge_index
        batch = data.batch

        node_features_1 = self.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)

        # num_nodes = node_features_2.size()[0]
        node_feature = node_features_2
        all_adj = to_dense_adj(edges)[0]

        all_kl_loss = 0
        all_pos_penalty = 0
        all_preserve_rate = 0
        all_graph_embedding = []
        all_noisy_embedding = []

        st = 0
        end = 0
        max_id = torch.max(batch)

        for i in range(int(max_id + 1)):
            
            j = 0
            while batch[st + j] == i and st + j <= len(batch) - 2:
                j += 1

            end = st + j

            if end == len(batch) - 1:
                end += 1
            
            one_batch_x = node_feature[st:end]
            num_nodes = one_batch_x.size(0)
            
            # this part is used to add noise
            static_node_feature = one_batch_x.clone().detach()
            node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim = 0)

            # this part is used to generate assignment matrix
            abstract_features_1 = torch.tanh(self.fully_connected_1(one_batch_x))
            assignment = F.softmax(self.fully_connected_2(abstract_features_1), dim = 1)
            gumbel_assignment = self.gumbel_softmax(assignment)

            # graph embedding
            graph_feature = torch.sum(one_batch_x, dim = 0, keepdim = True)

            # add noise to the node representation
            node_feature_mean = node_feature_mean.repeat(num_nodes, 1)

            # noisy graph representation
            lambda_pos = gumbel_assignment[:, 0].unsqueeze(dim = 1)
            lambda_neg = gumbel_assignment[:, 1].unsqueeze(dim = 1)

            noisy_node_feature_mean = lambda_pos * one_batch_x + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std

            noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
            noisy_graph_feature = torch.sum(noisy_node_feature, dim = 0, keepdim = True)

            KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std+epsilon) ** 2) + \
                torch.sum(((noisy_node_feature_mean -node_feature_mean)/(node_feature_std + epsilon))**2, dim = 0)
            KL_loss = torch.mean(KL_tensor)

            if torch.cuda.is_available():
                EYE = torch.ones(2).cuda()
                Pos_mask = torch.FloatTensor([1, 0]).cuda()
            else:
                EYE = torch.ones(2)
                Pos_mask = torch.FloatTensor([1, 0])

            Adj = all_adj[st:end,st:end]
            Adj.requires_grad = False
            new_adj = torch.mm(torch.t(assignment), Adj)
            new_adj = torch.mm(new_adj, assignment)

            normalize_new_adj = F.normalize(new_adj, p = 1, dim = 1)
            norm_diag = torch.diag(normalize_new_adj)
            pos_penalty = self.mseloss(norm_diag, EYE)
            
            # cal preserve rate (?)
            preserve_rate = torch.sum(assignment[:, 0] > 0.5) / assignment.size(0)

            all_kl_loss = all_kl_loss + KL_loss
            all_pos_penalty = all_pos_penalty + pos_penalty
            all_preserve_rate = all_preserve_rate + preserve_rate

            all_graph_embedding.append(graph_feature)
            all_noisy_embedding.append(noisy_graph_feature)
            
            st = end

        all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim = 0)
        all_noisy_embedding = torch.cat(tuple(all_noisy_embedding), dim = 0)
        all_pos_penalty = all_pos_penalty / (max_id + 1)
        all_kl_loss = all_kl_loss / (max_id + 1)
        all_preserve_rate = all_preserve_rate / (max_id + 1)
        
        return all_graph_embedding, all_noisy_embedding, all_pos_penalty, all_kl_loss, all_preserve_rate

    # def forward(self, data):
    #     epsilon = 0.0000001
        
    #     features = data.x
    #     edges = data.edge_index

    #     node_features_1 = self.relu(self.graph_convolution_1(features, edges))
    #     node_features_2 = self.graph_convolution_2(node_features_1, edges)
    #     num_nodes = node_features_2.size()[0]

    #     # this part is used to add noise
    #     node_feature = node_features_2
    #     static_node_feature = node_feature.clone().detach()
    #     node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim = 0)

    #     # this part is used to generate assignment matrix
    #     abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature))
    #     assignment = F.softmax(self.fully_connected_2(abstract_features_1), dim = 1)
    #     gumbel_assignment = self.gumbel_softmax(assignment)

    #     # graph embedding
    #     readout = get_readout_layers('mean')[0]
    #     graph_feature = readout(node_feature, data.batch)

    #     # add noise to the node representation
    #     node_feature_mean = node_feature_mean.repeat(num_nodes, 1)

    #     # noisy graph representation
    #     lambda_pos = gumbel_assignment[:, 0].unsqueeze(dim = 1)
    #     lambda_neg = gumbel_assignment[:, 1].unsqueeze(dim = 1)
    #     subgraph_representation = readout(lambda_pos * node_feature, data.batch)

    #     noisy_node_feature_mean = lambda_pos * node_feature + lambda_neg * node_feature_mean
    #     noisy_node_feature_std = lambda_neg * node_feature_std

    #     noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
    #     noisy_graph_feature = readout(noisy_node_feature, data.batch)

    #     KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std+epsilon) ** 2) + \
    #         torch.sum(((noisy_node_feature_mean -node_feature_mean)/(node_feature_std + epsilon))**2, dim = 0)
    #     KL_loss = torch.mean(KL_tensor)

    #     if torch.cuda.is_available():
    #         EYE = torch.ones(2).cuda()
    #         Pos_mask = torch.FloatTensor([1, 0]).cuda()
    #     else:
    #         EYE = torch.ones(2)
    #         Pos_mask = torch.FloatTensor([1, 0])

    #     Adj = to_dense_adj(edges)[0]
    #     Adj.requires_grad = False
    #     new_adj = torch.mm(torch.t(assignment), Adj)
    #     new_adj = torch.mm(new_adj, assignment)

    #     normalize_new_adj = F.normalize(new_adj, p = 1, dim = 1)
    #     norm_diag = torch.diag(normalize_new_adj)
    #     pos_penalty = self.mseloss(norm_diag, EYE)

    #     # cal preserve rate (?)
    #     preserve_rate = torch.sum(assignment[:, 0] > 0.5) / assignment.size(0)
        
    #     return graph_feature, noisy_graph_feature, subgraph_representation, pos_penalty, KL_loss, preserve_rate


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        
        self.args = args
        self.lin1 = nn.Linear(self.args.second_gcn_dimensions, self.args.cls_hidden_dimensions)
        self.lin2 = nn.Linear(self.args.cls_hidden_dimensions, 1)
        self.relu = nn.ReLU()

        # self.mlp = nn.ModuleList()
        # self.mlp.append(
        #     nn.Linear(self.args.second_gcn_dimensions, self.args.cls_hidden_dimensions), 
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.args.cls_hidden_dimensions, 1)
        # )
    
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        
    def forward(self, data):
        out = self.lin1(data)
        out = self.relu(out)
        out = self.lin2(out)
        
        return out
