
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader


def cross_validation_with_val_set(dataset, model, classifier, device, args, logger=None):
    
    val_losses, accs, durations = [], [], []
    
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.seed))):
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
        
        model.to(device).reset_parameters()
        classifier.to(device).reset_parameters()
        
        model_param_group = []
        model_param_group.append({"params": model.parameters()})
        model_param_group.append({"params": classifier.parameters()})
        optimizer = Adam(model_param_group, lr = args.learning_rate, weight_decay = args.weight_decay)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t_start = time.perf_counter()
        
        for epoch in range(1, args.epochs+1):
            train_loss = train(model, classifier, optimizer, device, train_loader, args)
            
            if train_loss != train_loss:
                print('NaN')
                continue
            
            val_losses.append(eval_loss(model, classifier, device, val_loader))
            accs.append(eval_acc(model, classifier, device, test_loader))
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
    print(acc)
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std


def train(model, classifier, optimizer, device, loader, args):
    model.train()
    
    total_loss = 0
    
    for data in loader: 
        data = data.to(device)
        num_graphs = len(data.y)
        
        embedding, noisy, pos_penalty, kl_loss, preserve_rate = model(data)
        features = torch.cat((embedding, noisy), dim = 0)
        labels = torch.cat((data.y, data.y), dim = 0)
        if torch.cuda.is_available():
            labels = labels.cuda()
        
        pred = classifier(features)
        cls_loss = nn.BCEWithLogitsLoss(reduction = 'mean')(pred, labels.to(torch.float).view(pred.shape))
        mi_loss = kl_loss / num_graphs
        
        optimizer.zero_grad()
        loss = cls_loss + args.con_weight * pos_penalty
        loss = loss + args.mi_weight * mi_loss
        loss.backward()
        
        total_loss += loss.item() * num_graphs
        optimizer.step()
    
    return total_loss / len(loader.dataset)


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


def eval_acc(model, classifier, device, loader):
    model.eval()
    classifier.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred,_,_,_,_ = model(data)
            pred = torch.nn.functional.sigmoid(classifier(pred).view(-1))
            pred = torch.where(pred>0.5, 1, 0)
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, classifier, device, loader):
    model.eval()
    classifier.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out,_,_,_,_ = model(data)
            out = classifier(out)
        loss += nn.BCEWithLogitsLoss(reduction = 'sum')(out.view(-1), data.y.to(torch.float32)).item()
    return loss / len(loader.dataset)
