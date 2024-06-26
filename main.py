import argparse
import numpy as np
from copy import deepcopy

import torch
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from utils import set_seed, save_model
from dataset import load_dataset

# GIB
from gib.arguments import load_gib_args
from gib.model import GIBGIN, Discriminator
from gib.train import gib_train, gib_eval

# VGIB
from vgib.arguments import load_vgib_args
from vgib.model import VariationalGIB, Classifier
from vgib.train import vgib_train, vgib_eval


def gib_main():
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}', '\n')
    
    gib_args = load_gib_args()
    dataset = load_dataset('dataset', args.dataset)

    path = f'saved_model/{args.model}/{args.dataset}'
    file_name = f'{args.model}_{args.dataset}'
    
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    gib_params_list, gib_disc_params_list = [], []
    gib_optim_params_list, gib_disc_optim_params_list = [], []
    
    for seed in range(args.num_runs):
        print(f'======================= Run: {seed} =================')
        set_seed(seed)    
        
        num_train = int(len(dataset) * args.train_frac)
        num_val = int(len(dataset) * args.val_frac)
        num_test = len(dataset) - num_train - num_val
        
        train, val, test = random_split(dataset, lengths = [num_train, num_val, num_test], generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train, batch_size = 128, shuffle = True)
        val_loader = DataLoader(val, batch_size = 128, shuffle = False)
        test_loader = DataLoader(test, batch_size = 128, shuffle = False)

        gib = GIBGIN(dataset, gib_args.num_layers, gib_args.hidden).to(device)
        gib_discriminator = Discriminator(gib_args.hidden).to(device)

        optimizer = Adam(gib.parameters(), lr = gib_args.lr)
        optimizer_local = Adam(gib_discriminator.parameters(), lr = gib_args.lr)

        best_val_loss, best_val_acc = 0, 0
        final_test_loss, final_test_acc = 0, 0
        for epoch in range(1, gib_args.epochs + 1):
            train_loss = gib_train(gib, gib_discriminator, optimizer, optimizer_local, device, train_loader, gib_args)
            val_loss, val_acc = gib_eval(gib, device, val_loader)
            test_loss, test_acc = gib_eval(gib, device, test_loader)
            
            if (val_acc > best_val_acc) or ((val_loss < best_val_loss) and (val_acc == best_val_acc)):
                best_val_loss = val_loss
                best_val_acc = val_acc
                final_test_loss = test_loss
                final_test_acc = test_acc
                
                gib_params = deepcopy(gib.state_dict())
                gib_disc_params = deepcopy(gib_discriminator.state_dict())
                gib_optim_params = deepcopy(optimizer.state_dict())
                gib_disc_optim_params = deepcopy(optimizer_local.state_dict())
            
            print(f'=== epoch: {epoch}')
            print(f'Train loss: {train_loss:.5f} | ', \
                f'Validation loss: {val_loss:.5f}, Acc: {val_acc:.5f} | ', \
                f'Test loss: {test_loss:.5f}, Acc: {test_acc:.5f}')
        
        val_losses.append(best_val_loss)
        val_accs.append(best_val_acc)
        test_losses.append(final_test_loss)
        test_accs.append(final_test_acc)
        
        gib_params_list.append(gib_params)
        gib_disc_params_list.append(gib_disc_params)
        gib_optim_params_list.append(gib_optim_params)
        gib_disc_optim_params_list.append(gib_disc_optim_params)

    best_idx = val_accs.index(max(val_accs))
    checkpoints = {
        'gib_params_dict': gib_params_list[best_idx],
        'gib_disc_params_dict': gib_disc_params_list[best_idx],
        'gib_optim_params_dict': gib_optim_params_list[best_idx],
        'gib_disc_optim_params_dict': gib_disc_optim_params_list[best_idx]
    }
    save_model(checkpoints, path, file_name)
    
    print('')
    print(f'Test Performance: {np.mean(test_accs):.3f} ({np.std(test_accs):.3f})')


def vgib_main():
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}', '\n')

    vgib_args = load_vgib_args()
    dataset = load_dataset('dataset', args.dataset)
    
    path = f'saved_model/{args.model}/{args.dataset}'
    file_name = f'{args.model}_{args.dataset}'

    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    vgib_params_list, vgib_cls_params_list = [], []
    vgib_optim_params_list = []
    
    for seed in range(args.num_runs):
        print(f'======================= Run: {seed} =================')
        set_seed(seed)    
        
        num_train = int(len(dataset) * args.train_frac)
        num_val = int(len(dataset) * args.val_frac)
        num_test = len(dataset) - num_train - num_val
        
        train, val, test = random_split(dataset, lengths = [num_train, num_val, num_test], generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train, batch_size = 128, shuffle = True)
        val_loader = DataLoader(val, batch_size = 128, shuffle = False)
        test_loader = DataLoader(test, batch_size = 128, shuffle = False)

        vgib = VariationalGIB(vgib_args, dataset.num_features).to(device)
        vgib_classifier = Classifier(vgib_args, dataset.num_classes).to(device)

        optimizer = Adam(list(vgib.parameters()) + list(vgib_classifier.parameters()), lr = vgib_args.lr)

        best_val_loss, best_val_acc = 0, 0
        final_test_loss, final_test_acc = 0, 0
        for epoch in range(1, vgib_args.epochs + 1):
            train_loss = vgib_train(vgib, vgib_classifier, optimizer, device, train_loader, vgib_args)
            val_loss, val_acc = vgib_eval(vgib, vgib_classifier, device, val_loader)
            test_loss, test_acc = vgib_eval(vgib, vgib_classifier, device, test_loader)
            
            if (val_acc > best_val_acc) or ((val_loss < best_val_loss) and (val_acc == best_val_acc)):
                best_val_loss = val_loss
                best_val_acc = val_acc
                final_test_loss = test_loss
                final_test_acc = test_acc
                
                vgib_params = deepcopy(vgib.state_dict())
                vgib_cls_params = deepcopy(vgib_classifier.state_dict())
                vgib_optim_params = deepcopy(optimizer.state_dict())
            
            print(f'=== epoch: {epoch}')
            print(f'Train loss: {train_loss:.5f} | ', \
                f'Validation loss: {val_loss:.5f}, Acc: {val_acc:.5f} | ', \
                f'Test loss: {test_loss:.5f}, Acc: {test_acc:.5f}')
        
        val_losses.append(best_val_loss)
        val_accs.append(best_val_acc)
        test_losses.append(final_test_loss)
        test_accs.append(final_test_acc)
        
        vgib_params_list.append(vgib_params)
        vgib_cls_params_list.append(vgib_cls_params)
        vgib_optim_params_list.append(vgib_optim_params)
    
    best_idx = val_accs.index(max(val_accs))
    checkpoints = {
        'vgib_params_dict': vgib_params_list[best_idx],
        'vgib_cls_params_dict': vgib_cls_params_list[best_idx],
        'vgib_optim_params_dict': vgib_optim_params_list[best_idx]
    }
    save_model(checkpoints, path, file_name)

    print('')
    print(f'Test Performance: {np.mean(test_accs):.3f} ({np.std(test_accs):.3f})')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'GIB')
    parser.add_argument('--dataset', type = str, default = 'MUTAG')
    parser.add_argument('--num_runs', type = int, default = 10)
    parser.add_argument('--train_frac', type = float, default = 0.8)
    parser.add_argument('--val_frac', type = float, default = 0.1)
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    print(args)
    
    if args.model == 'GIB':
        gib_main()
    elif args.model == 'VGIB':
        vgib_main()
