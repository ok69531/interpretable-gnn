import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.backends
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from module.utils import set_seed, save_model
from module.load_dataset import get_dataset

# GIB
from gib.arguments import load_gib_args
from gib.model import GIBGIN, Discriminator
from gib.train import gib_train, gib_eval

# VGIB
from vgib.arguments import load_vgib_args
from vgib.model import VariationalGIB, Classifier
from vgib.train import vgib_train, vgib_eval

# GSAT
from gsat.arguments import load_gsat_args
from gsat.model import GIN, ExtractorMLP, GSAT
from gsat.train import run_one_epoch

# PGIB
from pgib.arguments import load_pgib_args
from pgib.model import PGIBGIN
from pgib.train import pgib_train, pgib_evaluate_GC, pgib_test_GC
from pgib.my_mcts import mcts
from pgib.proto_join import join_prototypes_by_activations


def gib_main(args, device):
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}', '\n')
    
    gib_args = load_gib_args()
    dataset = get_dataset('dataset', args.dataset)

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


def vgib_main(args, device):
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}', '\n')

    vgib_args = load_vgib_args()
    dataset = get_dataset('dataset', args.dataset)
    
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


def gsat_main(args, device):
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}', '\n')

    gsat_args = load_gsat_args()
    if args.dataset == 'SMotif':
        gsat_args.learn_edge_att = True
        gsat_args.lr = 0.003
        gsat_args.final_r = 0.7
    dataset = get_dataset('dataset', args.dataset)
    
    path = f'saved_model/{args.model}/{args.dataset}'
    file_name = f'{args.model}_{args.dataset}'

    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    gsat_params_list = []
    gsat_optim_params_list = []
    
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

        num_class = dataset.num_classes
        gnn = GIN(dataset.num_features, num_class, gsat_args).to(device)
        extractor = ExtractorMLP(gsat_args).to(device)
        lr, wd = gsat_args.lr, gsat_args.weight_decay
        optimizer = torch.optim.Adam(list(extractor.parameters()) + list(gnn.parameters()), lr=lr, weight_decay=wd)
        gsat = GSAT(gnn, extractor, optimizer, device, num_class, gsat_args)

        best_val_loss, best_val_acc = 0, 0
        final_test_loss, final_test_acc = 0, 0
        for epoch in range(1, gsat_args.epochs + 1):
            train_loss, _, _, _ = run_one_epoch(gsat, train_loader, epoch, 'train', device, gsat_args)
            val_loss, val_acc, _, _ = run_one_epoch(gsat, val_loader, epoch, 'valid', device, gsat_args)
            test_loss, test_acc, _, _ = run_one_epoch(gsat, test_loader, epoch, 'valid', device, gsat_args)
            
            if (val_acc > best_val_acc) or ((val_loss < best_val_loss) and (val_acc == best_val_acc)):
                best_val_loss = val_loss
                best_val_acc = val_acc
                final_test_loss = test_loss
                final_test_acc = test_acc
                
                gsat_params = deepcopy(gsat.state_dict())
                gsat_optim_params = deepcopy(gsat.optimizer.state_dict())
            
            print(f'=== epoch: {epoch}')
            print(f'Train loss: {train_loss:.5f} | ', \
                f'Validation loss: {val_loss:.5f}, Acc: {val_acc:.5f} | ', \
                f'Test loss: {test_loss:.5f}, Acc: {test_acc:.5f}')
        
        val_losses.append(best_val_loss)
        val_accs.append(best_val_acc)
        test_losses.append(final_test_loss)
        test_accs.append(final_test_acc)
        
        gsat_params_list.append(gsat_params)
        gsat_optim_params_list.append(gsat_optim_params)
    
    best_idx = val_accs.index(max(val_accs))
    checkpoints = {
        'gsat_params_dict': gsat_params_list[best_idx],
        'gsat_optim_params_dict': gsat_optim_params_list[best_idx]
    }
    save_model(checkpoints, path, file_name)

    print('')
    print(f'Test Performance: {np.mean(test_accs):.3f} ({np.std(test_accs):.3f})')


def pgib_main(args, device):
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}', '\n')

    pgib_args = load_pgib_args()
    dataset = get_dataset('dataset', args.dataset)
    
    path = f'saved_model/{args.model}/{args.dataset}'
    file_name = f'{args.model}_{args.dataset}'

    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    pgib_params_list = []
    pgib_optim_params_list = []
    
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

        num_class = dataset.num_classes
        output_dim = int(dataset.num_classes)
        pgib = PGIBGIN(dataset.num_features, output_dim, pgib_args, cont = args.pgib_cont).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(pgib.parameters(), lr = pgib_args.lr, weight_decay = pgib_args.weight_decay)

        best_val_loss, best_val_acc = 0, 0
        final_test_loss, final_test_acc = 0, 0
        for epoch in range(1, pgib_args.epochs + 1):
            if epoch >= pgib_args.proj_epochs and epoch % 50 == 0:
                pgib.eval()
                
                for i in range(pgib.prototype_vectors.shape[0]):
                    count = 0
                    best_similarity = 0
                    label = pgib.prototype_class_identity[0].max(0)[1]
                    # label = model.prototype_class_identity[i].max(0)[1]
                    
                    for j in range(i*10, len(train.indices)):
                        data = dataset[train.indices[j]]
                        if data.y == label:
                            count += 1
                            coalition, similarity, prot = mcts(data, pgib, pgib.prototype_vectors[i])
                            pgib.to(device)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                proj_prot = prot
                        
                        if count >= pgib_args.count:
                            pgib.prototype_vectors.data[i] = proj_prot.to(device)
                            print('Projection of prototype completed')
                            break

                # prototype merge
                share = True
                if pgib_args.share: 
                    if pgib.prototype_vectors.shape[0] > round(output_dim * pgib_args.num_prototypes_per_class * (1-pgib_args.merge_p)) :  
                        join_info = join_prototypes_by_activations(pgib, pgib_args.proto_percnetile, train_loader, device, cont = args.pgib_cont)

            train_loss, _, _ = pgib_train(pgib, optimizer, device, train_loader, criterion, epoch, pgib_args, cont = args.pgib_cont)
            
            if train_loss != train_loss:
                print('Train loss is NaN.')
                break
            
            val_eval_dict = pgib_evaluate_GC(val_loader, pgib, device, criterion)
            val_loss, val_acc = val_eval_dict['loss'], val_eval_dict['acc']
            test_eval_dict = pgib_test_GC(test_loader, pgib, device, criterion)[0]
            test_loss, test_acc = test_eval_dict['loss'], test_eval_dict['acc']
            
            if (val_acc > best_val_acc) or ((val_loss < best_val_loss) and (val_acc == best_val_acc)):
                best_val_loss = val_loss
                best_val_acc = val_acc
                final_test_loss = test_loss
                final_test_acc = test_acc
                
                pgib_params = deepcopy(pgib.state_dict())
                pgib_optim_params = deepcopy(optimizer.state_dict())
            
            print(f'=== epoch: {epoch}')
            print(f'Train loss: {train_loss:.5f} | ', \
                f'Validation loss: {val_loss:.5f}, Acc: {val_acc:.5f} | ', \
                f'Test loss: {test_loss:.5f}, Acc: {test_acc:.5f}')
        
        val_losses.append(best_val_loss)
        val_accs.append(best_val_acc)
        test_losses.append(final_test_loss)
        test_accs.append(final_test_acc)
        
        pgib_params_list.append(pgib_params)
        pgib_optim_params_list.append(pgib_optim_params)
    
    best_idx = val_accs.index(max(val_accs))
    checkpoints = {
        'gsat_params_dict': pgib_params_list[best_idx],
        'gsat_optim_params_dict': pgib_optim_params_list[best_idx]
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
    parser.add_argument('--pgib_cont', type = bool, default = False)
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    print(args)
    
    if args.model == 'GIB':
        gib_main(args, device)
    elif args.model == 'VGIB':
        vgib_main(args, device)
    elif args.model == 'GSAT':
        gsat_main(args, device)
    elif args.model == 'PGIB':
        pgib_main(args, device)
