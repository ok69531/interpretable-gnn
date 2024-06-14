# https://github.com/Samyu0304/graph-information-bottleneck-for-Subgraph-Recognition/tree/main

import argparse

import torch
import torch.backends
from torch_geometric.datasets import TUDataset

from train import cross_validation_with_val_set
from gin import GIBGIN, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--split_seed', type = int, default = 12345)
parser.add_argument('--dataset', type = str, default = 'MUTAG')
parser.add_argument('--num_layers', type = int, default = 2)
parser.add_argument('--hidden', type = int, default = 16)
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--lr_decay_factor', type = float, default = 0.5)
parser.add_argument('--lr_decay_step_size', type = int, default = 50)
parser.add_argument('--inner_loop', type = int, default = 50)
parser.add_argument('--beta', type = float, default = 0.1)
parser.add_argument('--pp_weight', type = float, default = 0.3)
parser.add_argument('--folds', type = int, default = 10)
args = parser.parse_args([])


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))


torch.manual_seed(args.seed)

path = '../dataset'
dataset = TUDataset(path, args.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = GIBGIN(dataset, args.num_layers, args.hidden)
discriminator = Discriminator(args.hidden)

results = []
best_result = (float('inf'), 0, 0)  # (loss, acc, std)

loss, acc, std = cross_validation_with_val_set(
    dataset,
    model,
    discriminator,
    device,
    args,
    logger= None
)
if loss < best_result[0]:
    best_result = (loss, acc, std)

desc = '{:.3f} , {:.3f}'.format(best_result[1], best_result[2])
print('Best result - {}'.format(desc))
results += ['{} - {}: {}'.format(args.dataset, model, desc)]
