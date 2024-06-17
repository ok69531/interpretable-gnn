import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from itertools import accumulate
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

from gin import PGIBGIN
from my_mcts import mcts
from proto_join import join_prototypes_by_activations


def k_fold(dataset, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

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


def warm_only(model):
    for p in model.gnn_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.gnn_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True


def train(model, optimizer, device, loader, criterion, epoch, args):
    model.train()
    
    if epoch < args.warm_epochs:
        warm_only(model)
    else:
        joint(model)

    acc = []
    loss_list = []
    ld_loss_list = []

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        
        if args.cont:
            logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = model(batch)
        else:
            logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = model(batch)

        cls_loss = criterion(logits, batch.y)
        
        if args.cont:
            prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, batch.y]).to(device)
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            positive_sim_matrix = sim_matrix * prototypes_of_correct_class
            negative_sim_matrix = sim_matrix * prototypes_of_wrong_class
            
            contrastive_loss = positive_sim_matrix.sum(dim = 1) / negative_sim_matrix.sum(dim = 1)
            contrastive_loss = - torch.log(contrastive_loss).mean()
        
        prototype_numbers = []
        for i in range(model.prototype_class_identity.shape[1]):
            prototype_numbers.append(int(torch.count_nonzero(model.prototype_class_identity[:, i])))
        prototype_numbers = accumulate(prototype_numbers)
        
        n = 0
        ld = 0
        
        for k in prototype_numbers:
            p = model.prototype_vectors[n:k]
            n = k
            p = F.normalize(p, p = 2, dim = 1)
            matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(device) - 0.3
            matrix2 = torch.zeros(matrix1.shape).to(device)
            ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))
            
        if args.cont:
            loss = cls_loss + args.alpha2 * contrastive_loss + args.con_weight * connectivity_loss + args.alpha1 * KL_Loss
        else:
            loss = cls_loss + args.alpha2 * prototype_pred_loss + args.con_weight * connectivity_loss + args.alpha1 * KL_Loss

        # optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value = 2.0)
        optimizer.step()
        
        # reocrd
        _, prediction = torch.max(logits, -1)
        loss_list.append(loss.item())
        ld_loss_list.append(ld.item())
        acc.append(prediction.eq(batch.y).cpu().numpy())

    # report train msg
    # print(f'Train Epoch: {epoch} | Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | '
    #     f'Acc: {np.concatenate(acc, axis = 0).mean():.3f}')
    
    return np.average(loss_list), np.average(ld_loss_list), np.concatenate(acc, axis = 0).mean()


def cross_validation_with_val_set(dataset, device, args, logger=None):
    
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    
    val_losses, accs, durations = [], [], []
    
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.split_seed))):
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        
        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, args.batch_size, shuffle = True)
            val_loader = DenseLoader(val_dataset, args.batch_size, shuffle = False)
            test_loader = DenseLoader(test_dataset, args.batch_size, shuffle = False)
        else:
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle = True)
            val_loader = DataLoader(val_dataset, args.batch_size, shuffle = False)
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle = False)
        
        model = PGIBGIN(input_dim, output_dim, args)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        best_acc = 0.
        early_stop_count = 0
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t_start = time.perf_counter()

        for epoch in range(args.epochs):
            acc = []
            loss_list = []
            ld_loss_list = []

            if epoch >= args.proj_epochs and epoch % 50 == 0:
                model.eval()
                
                for i in range(model.prototype_vectors.shape[0]):
                    count = 0
                    best_similarity = 0
                    label = model.prototype_class_identity[0].max(0)[1]
                    # label = model.prototype_class_identity[i].max(0)[1]
                    
                    for j in range(i*10, len(train_idx)):
                        data = dataset[train_idx[j]]
                        if data.y == label:
                            count += 1
                            coalition, similarity, prot = mcts(data, model, model.prototype_vectors[i])
                            model.to(device)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                proj_prot = prot
                        
                        if count >= args.count:
                            model.prototype_vectors.data[i] = proj_prot.to(device)
                            print('Projection of prototype completed')
                            break

                # prototype merge
                share = True
                if args.share: 
                    if model.prototype_vectors.shape[0] > round(output_dim * args.num_prototypes_per_class * (1-args.merge_p)) :  
                        join_info = join_prototypes_by_activations(model, args.proto_percnetile, train_loader, device)


            train_loss, _, _ = train(model, optimizer, device, train_loader, criterion, epoch, args)
            
            if train_loss != train_loss:
                print('NaN')
                continue

            val_losses.append(evaluate_GC(val_loader, model, device, criterion)['loss'])
            accs.append(test_GC(test_loader, model, device, criterion)[0]['acc'])
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss, 
                'val_loss': val_losses[-1],
                'test_acc': accs[-1]
            }
            
            print(eval_info)
            
            if logger is not None:
                logger(eval_info)
            
            # if epoch % args.lr_decay_step_size == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = args.lr_decay_factor * param_group['lr']
            
        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        
    loss, acc, duration = torch.tensor(val_losses), torch.tensor(accs), torch.tensor(durations)
    loss, acc = loss.view(args.folds, args.epochs), acc.view(args.folds, args.epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(args.folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std


def evaluate_GC(loader, model, device, criterion):
    model.eval()
    
    acc = []
    loss_list = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, probs, _, _, _, _, _, _ = model(batch)
            loss = criterion(logits, batch.y)
            
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
        
        eval_state = {
            'loss': np.average(loss_list),
            'acc': np.concatenate(acc, axis = 0).mean()
        }
    
    return eval_state


def test_GC(loader, model, device, criterion):
    model.eval()
    
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    
    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = batch.to(device)
            logits, probs, active_node_index, _, _, _, _, _ = model(batch)
            loss = criterion(logits, batch.y)
            
            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)
    
    test_state = {
        'loss': np.average(loss_list),
        'acc': np.average(np.concatenate(acc, axis=0).mean())
    }
    
    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    
    return test_state, pred_probs, predictions
