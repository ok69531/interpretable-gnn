import argparse


def load_pgib_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--split_seed', type = int, default = 42)
    parser.add_argument('--dataset', type = str, default = 'MUTAG')
    parser.add_argument('--latent_dim', type = list, default = [128, 128, 128])
    parser.add_argument('--batch_size', type = int, default = 24)
    parser.add_argument('--epochs', type = int, default = 300)
    parser.add_argument('--num_prototypes_per_class', type = int, default = 7)
    parser.add_argument('--readout', type = str, default = 'max')
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
    
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    return args