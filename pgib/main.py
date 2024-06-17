import argparse

import torch
import torch.backends
import torch.nn as nn
from torch_geometric.datasets import TUDataset

from train import cross_validation_with_val_set


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--split_seed', type = int, default = 42)
parser.add_argument('--dataset', type = str, default = 'MUTAG')
parser.add_argument('--latent_dim', type = list, default = [128, 128, 128])
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--epochs', type = int, default = 300)
parser.add_argument('--num_prototypes_per_class', type = int, default = 7)
parser.add_argument('--readout', type = str, default = 'sum')
parser.add_argument('--cont', type = bool, default = True)
parser.add_argument('--lr', type = float, default = 0.005)
parser.add_argument('--weight_decay', type = float, default = 0)
parser.add_argument('--beta', type = float, default = 0.1)
parser.add_argument('--pp_weight', type = float, default = 0.3)
parser.add_argument('--folds', type = int, default = 10)
parser.add_argument('--warm_epochs', type = int, default = 10)
parser.add_argument('--alpha1', type = float, default = 0.0001)
parser.add_argument('--alpha2', type = float, default = 0.01)
parser.add_argument('--con_weight', type = int, default = 5)
parser.add_argument('--early_stopping', type = int, default = 10000)
parser.add_argument('--proj_epochs', type = int, default = 50)
parser.add_argument('--count', type = int, default = 1)
parser.add_argument('--share', type = bool, default = True)
parser.add_argument('--merge_p', type = float, default = 0.3)
parser.add_argument('--proto_percnetile', type = float, default = 0.1)
args = parser.parse_args([])


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))


path = '../dataset'
dataset = TUDataset(path, args.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)

avg_nodes = 0.
avg_edge_index = 0.
for i in range(len(dataset)):
    avg_nodes += dataset[i].x.shape[0]
    avg_edge_index += dataset[i].edge_index.shape[1]
avg_nodes /= len(dataset)
avg_edge_index /= len(dataset)

print(f'Datset : {args.dataset}')
print(f'graphs {len(dataset)}, avg_nodes {avg_nodes:.4f}, avg_edge index {avg_edge_index/2:.4f}')


results = []
best_result = (float('inf'), 0, 0)  # (loss, acc, std)

loss, acc, std = cross_validation_with_val_set(
    dataset,
    device,
    args,
    logger= None
)
if loss < best_result[0]:
    best_result = (loss, acc, std)

desc = '{:.3f} , {:.3f}'.format(best_result[1], best_result[2])
print('Best result - {}'.format(desc))
results += ['{} : {}'.format(args.dataset, desc)]
# results += ['{} - {}: {}'.format(args.dataset, model, desc)]