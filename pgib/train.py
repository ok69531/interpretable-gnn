import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from itertools import accumulate
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader


def warm_only(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def pgib_train(model, optimizer, device, loader, criterion, epoch, args, cont):
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
        
        if cont:
            logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = model(batch)
        else:
            logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = model(batch)

        cls_loss = criterion(logits, batch.y)
        
        if cont:
            prototypes_of_correct_class = torch.t(model.model.prototype_class_identity[:, batch.y]).to(device)
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            positive_sim_matrix = sim_matrix * prototypes_of_correct_class
            negative_sim_matrix = sim_matrix * prototypes_of_wrong_class
            
            contrastive_loss = positive_sim_matrix.sum(dim = 1) / negative_sim_matrix.sum(dim = 1)
            contrastive_loss = - torch.log(contrastive_loss).mean()
        
        prototype_numbers = []
        for i in range(model.model.prototype_class_identity.shape[1]):
            prototype_numbers.append(int(torch.count_nonzero(model.model.prototype_class_identity[:, i])))
        prototype_numbers = accumulate(prototype_numbers)
        
        n = 0
        ld = 0
        
        for k in prototype_numbers:
            p = model.model.prototype_vectors[n:k]
            n = k
            p = F.normalize(p, p = 2, dim = 1)
            matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(device) - 0.3
            matrix2 = torch.zeros(matrix1.shape).to(device)
            ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))
            
        if cont:
            loss = cls_loss + args.alpha2 * contrastive_loss + args.con_weight * connectivity_loss + args.alpha1 * KL_Loss
        else:
            loss = cls_loss + args.alpha2 * prototype_pred_loss + args.con_weight * connectivity_loss + args.alpha1 * KL_Loss

        # optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.model.parameters(), clip_value = 2.0)
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


def pgib_evaluate_GC(loader, model, device, criterion):
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


def pgib_test_GC(loader, model, device, criterion):
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
