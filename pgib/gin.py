import pdb
from itertools import accumulate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
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


class PGIBGIN(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(PGIBGIN, self).__init__()
        
        self.args = args
        self.output_dim = output_dim
        self.latent_dim = args.latent_dim
        self.num_gnn_layers = len(self.latent_dim)
        self.dense_dim = self.latent_dim[-1]
        self.num_prototypes_per_class = args.num_prototypes_per_class
        
        self.readout_layers = get_readout_layers(args.readout)
        
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, self.latent_dim[0], bias = False),
                    nn.ReLU(),
                    nn.Linear(self.latent_dim[0], self.latent_dim[0], bias = False),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.latent_dim[0])
                ), train_eps = False))

        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(self.latent_dim[i-1], self.latent_dim[i], bias = False),
                        nn.ReLU(),
                        nn.Linear(self.latent_dim[i], self.latent_dim[i], bias = False),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.latent_dim[1])
                    ), train_eps = False))

        self.fully_connected_1 = nn.Linear(self.latent_dim[2], self.latent_dim[2])
        self.fully_connected_2 = nn.Linear(self.latent_dim[2], 2)

        self.softmax = nn.Softmax(dim = -1)

        # prototype layers
        self.epsilon = 1e-4
        self.prototype_shape = (output_dim * self.num_prototypes_per_class, self.latent_dim[2])
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape, requires_grad = True))
        self.num_prototypes = self.prototype_shape[0]
        self.prototype_predictor = nn.Linear(self.latent_dim[2], self.num_prototypes * self.latent_dim[2], bias = False)
        self.mse_loss = nn.MSELoss()

        self.last_layer = nn.Linear(self.latent_dim[2] + self.num_prototypes, output_dim, bias = False)

        assert (self.num_prototypes % output_dim == 0)

        self.prototype_class_identity = torch.zeros(self.num_prototypes, output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.set_last_layer_incorrect_connection(incorrect_strength = -0.5)
    
    
    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight[:,: self.num_prototypes].data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)


    def gumbel_softmax(self, prob):
        return F.gumbel_softmax(prob, tau = 1, dim = -1)


    def prototype_distance(self, graph_emb):
        xp = torch.mm(graph_emb, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(graph_emb ** 2, dim = 1, keepdim = True) + \
            torch.t(torch.sum(self.prototype_vectors ** 2, dim = 1, keepdim = True))
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        
        return similarity, distance
    
    
    def forward(self, data, merge = False):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch

        node_features_1 = F.relu(self.gnn_layers[0](x, edge_index))
        node_features_2 = self.gnn_layers[1](node_features_1, edge_index)
        num_nodes = node_features_2.size(0)

        # this part is used to add noise
        node_feature = node_features_2
        node_emb = node_feature

        # this part is used to generate assignment matrix
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature))
        assignment = F.softmax(self.fully_connected_2(abstract_features_1), dim = 1)

        gumbel_assignment = self.gumbel_softmax(assignment)

        # noisty graph representation
        lambda_pos = gumbel_assignment[:, 0].unsqueeze(dim = 1)
        lambda_neg = gumbel_assignment[:, 1].unsqueeze(dim = 1)

        # this is the graph embedding
        active = lambda_pos > 0.5
        active = active.squeeze()

        active_node_index = []
        node_number = [0]
        for i in range(batch[-1] + 1):
            node_number.append(len(batch[batch == i]))
        node_number = list(accumulate(node_number))

        for j in range(len(node_number) - 1):
            active_node_index.append(active[node_number[j] : node_number[j+1]].nonzero().squeeze().tolist())

        # KL loss
        static_node_feature = node_feature.clone().detach()
        node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim = 0)
        node_feature_mean = node_feature_mean.repeat(num_nodes, 1)

        noisy_node_feature_mean = lambda_pos * node_feature + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std

        noisy_node_feature = noisy_node_feature_mean + \
            torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std

        for readout in self.readout_layers:
            noisy_graph_feature = readout(noisy_node_feature, batch)
        graph_emb = noisy_graph_feature

        epsilon = 0.0000001
        KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std+epsilon) ** 2) + \
            torch.sum(((noisy_node_feature_mean -node_feature_mean)/(node_feature_std + epsilon))**2, dim = 0)
        KL_Loss = torch.mean(KL_tensor)

        Adj = to_dense_adj(edge_index, max_num_nodes = assignment.shape[0])[0]
        Adj.requires_grad = False

        try:
            torch.mm(torch.t(assignment),Adj)
        except:
            pdb.set_trace()
        new_adj = torch.mm(torch.t(assignment), Adj)
        new_adj = torch.mm(new_adj, assignment)
        normalize_new_adj = F.normalize(new_adj, p = 1, dim = 1)
        norm_diag = torch.diag(normalize_new_adj)

        # if torch.cuda.is_available():
        #     EYE = torch.ones(2).cuda()
        # else:
        #     EYE = torch.ones(2)
        EYE = torch.ones(2).to(edge_index.device)

        pos_penalty = self.mse_loss(norm_diag, EYE)

        ## graph embedding
        prototype_activations, min_distance = self.prototype_distance(graph_emb)

        final_embedding = torch.cat((prototype_activations, graph_emb), dim = 1)
        logits = self.last_layer(final_embedding)
        probs = self.softmax(logits)

        if self.args.cont:
            return logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, prototype_activations, min_distance
        else:
            for i in range(graph_emb.shape[0]):
                predicted_prototype = self.prototype_predictor(torch.t(graph_emb[i])).reshape(-1, self.prototype_vectors.shape[1]) 
                if i == 0:
                    prototype_pred_losses = self.mse_loss(self.prototype_vectors, predicted_prototype).reshape(1)
                else:
                    prototype_pred_losses = torch.cat((prototype_pred_losses, self.mse_loss(self.prototype_vectors, predicted_prototype).reshape(1)))
            prototype_pred_loss = prototype_pred_losses.mean()

            return logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, prototype_pred_loss, min_distance 
