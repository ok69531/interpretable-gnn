# https://github.com/Samyu0304/graph-information-bottleneck-for-Subgraph-Recognition/tree/main

import argparse

import torch
from torch_geometric.datasets import TUDataset

from train import cross_validation_with_val_set
from gin import VariationalGIB, Classifier


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--split_seed', type = int, default = 42)
parser.add_argument('--dataset', type = str, default = 'MUTAG')
parser.add_argument('--folds', type = int, default = 10)
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument("--subgraph_const",
                    type = float,
                    default= 0.8 ,
                    help="Folder with training graph jsons.")
parser.add_argument("--first-gcn-dimensions",
                    type=int,
                    default=16,
                help="Filters (neurons) in 1st convolution. Default is 32.")
parser.add_argument("--second-gcn-dimensions",
                    type=int,
                    default=16,
                help="Filters (neurons) in 2nd convolution. Default is 16.")
parser.add_argument("--first-dense-neurons",
                    type=int,
                    default=16,
                help="Neurons in SAGE aggregator layer. Default is 16.")
parser.add_argument("--second-dense-neurons",
                    type=int,
                    default=2,
                help="SAGE attention neurons. Default is 8.")
parser.add_argument("--cls_hidden_dimensions",
                    type=int,
                    default= 4,
                    help="classifier hidden dims")
parser.add_argument("--mi_weight",
                    type=float,
                    default= 0.1,
                    help="classifier hidden dims")
parser.add_argument("--con_weight",
                    type=float,
                    default= 5,
                    help="classifier hidden dims")
parser.add_argument("--learning_rate",
                    type=float,
                    default=0.001,
                help="Learning rate. Default is 0.01.")
parser.add_argument("--weight_decay",
                    type=float,
                    default=5*10**-5,
                help="Adam weight decay. Default is 5*10^-5.")
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

model = VariationalGIB(args, dataset.num_features)
classifier = Classifier(args)

results = []
best_result = (float('inf'), 0, 0)  # (loss, acc, std)

loss, acc, std = cross_validation_with_val_set(
    dataset,
    model,
    classifier,
    device,
    args,
    logger= None
)
if loss < best_result[0]:
    best_result = (loss, acc, std)

desc = '{:.3f} , {:.3f}'.format(best_result[1], best_result[2])
print('Best result - {}'.format(desc))
results += ['{} - {}: {}'.format(args.dataset, model, desc)]
