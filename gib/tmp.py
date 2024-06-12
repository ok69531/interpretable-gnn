#%%
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset


#%%
seed = 0
path = '../dataset'
dataset = TUDataset(path, 'MUTAG')


#%%
''' Model '''
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge
from torch_geometric.utils import to_dense_adj

hidden = 16
num_layers = 2
# initial node embedding to generate subgraph
conv1 = GINConv(Sequential(Linear(dataset.num_features, hidden),
                           ReLU(),
                           Linear(hidden, hidden),
                           ReLU(),
                           BN(hidden)), train_eps = False)
convs = nn.ModuleList()
for i in range(num_layers - 1):
    convs.append(GINConv(Sequential(Linear(hidden, hidden),
                                    ReLU(),
                                    Linear(hidden, hidden),
                                    ReLU(),
                                    BN(hidden)), train_eps=False))

# mlp for subgraph generation
cluster1 = Linear(hidden, hidden)
cluster2 = Linear(hidden, 2)

lin1 = Linear(hidden, hidden)
lin2 = Linear(hidden, dataset.num_classes)


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        
        self.input_size = 2 * hidden_size
        self.hidden_size = hidden_size
        
        self.lin1 = Linear(self.input_size, self.hidden_size)
        self.lin2 = Linear(self.hidden_size, 1)
        self.relu = ReLU()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        
    def forward(self, graph_embeddings, subgraph_embeddings):
        cat_embeddings = torch.cat((graph_embeddings, subgraph_embeddings), dim = -1)
        
        pre = self.relu(self.lin1(cat_embeddings))
        pre = self.relu(self.lin2(pre))
        
        return pre



#%%
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

folds = 10
batch_size = 128

def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices

indices = k_fold(dataset, folds)

train_idx = indices[0][0]
test_idx = indices[1][0]
val_idx = indices[2][0]

train_dataset = dataset[train_idx]
test_dataset = dataset[test_idx]
val_dataset = dataset[val_idx]

torch.manual_seed(seed)
train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size, shuffle = False)


#%%
# training procedure (model forward part)
for data in train_loader: break

x, edge_index, batch = data.x, data.edge_index, data.batch

# node embedding
h = conv1(x, edge_index)
for conv in convs:
    h = conv(h, edge_index)

# generage subgraph (S)
assignment = torch.tanh(cluster1(h))
assignment = F.softmax(cluster2(assignment), dim = 1)

# aggregate
max_id = torch.max(batch)
EYE = torch.ones(2)
all_adj = to_dense_adj(edge_index)[0]

all_pos_penalty = 0
all_graph_embedding = []
all_pos_embedding = []

st = 0
end = 0

for i in range(int(max_id + 1)):
    j = 0
    while batch[st + j] == i and st + j <= len(batch) - 2:
        j += 1
    end = st + j

    if end == len(batch) - 1: end += 1

    one_batch_x = h[st:end]
    one_batch_assignment = assignment[st:end]

    group_features = torch.mm(torch.t(one_batch_assignment), one_batch_x)
    pos_embedding = group_features[0].unsqueeze(dim = 0) # S^T X: represetation of g_sub

    Adj = all_adj[st:end, st:end]
    new_adj = torch.mm(torch.t(one_batch_assignment), Adj)
    new_adj = torch.mm(new_adj, one_batch_assignment)
    normalize_new_adj = F.normalize(new_adj, p = 1, dim = 1)
    norm_diag = torch.diag(normalize_new_adj)
    mse_loss = nn.MSELoss()
    pos_penalty = mse_loss(norm_diag, EYE) # connectivity loss

    graph_embedding = torch.mean(one_batch_x, dim = 0, keepdim = True)
    # graph_embedding = torch.mean(h, dim = 0, keepdim = True)

    all_pos_embedding.append(pos_embedding)
    all_graph_embedding.append(graph_embedding)

    all_pos_penalty = all_pos_penalty + pos_penalty

    st = end

all_pos_embedding = torch.cat(tuple(all_pos_embedding), dim = 0)
all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim = 0)
all_pos_penalty = all_pos_penalty / (max_id + 1)

h = F.relu(lin1(all_pos_embedding))
h = F.dropout(h, p = 0.5)
h = lin2(h)
out = F.log_softmax(h, dim=-1)

cls_loss = F.nll_loss(out, data.y.view(-1))

inner_loop = 50
discriminator = Discriminator(hidden)

