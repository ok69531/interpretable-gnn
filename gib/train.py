# https://github.com/Samyu0304/graph-information-bottleneck-for-Subgraph-Recognition/tree/main

import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader


def cross_validation_with_val_set(dataset, model, discriminator, device, args, logger=None):
    
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
        optimizer = Adam(model.parameters(), lr = args.lr)
        
        discriminator.to(device).reset_parameters()
        optimizer_local = Adam(discriminator.parameters(), lr = args.lr)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t_start = time.perf_counter()
        
        for epoch in range(1, args.epochs+1):
            train_loss = train(model, discriminator, optimizer, optimizer_local, device, train_loader, args)
            
            if train_loss != train_loss:
                print('NaN')
                continue
            
            val_losses.append(eval_loss(model, device, val_loader))
            accs.append(eval_acc(model, device, test_loader))
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
            
            if epoch % args.lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr_decay_factor * param_group['lr']
            
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


def train(model, discriminator, optimizer, local_optimizer, device, loader, args):
    model.train()
    
    total_loss = 0
    
    for data in loader: 
        data = data.to(device)
        out, all_sub_embedding, all_graph_embedding, all_con_penalty = model(data)
        
        # to find phi_2^*
        for j in range(0, args.inner_loop):
            local_optimizer.zero_grad()
            local_loss = -MI_Est(discriminator, all_graph_embedding.detach(), all_sub_embedding.detach())
            local_loss.backward()
            local_optimizer.step()
        
        optimizer.zero_grad()
        
        cls_loss = F.nll_loss(out, data.y.view(-1))
        mi_loss = MI_Est(discriminator, all_graph_embedding, all_sub_embedding)
        loss = (1 - args.pp_weight) * (cls_loss + args.beta * mi_loss) + args.pp_weight * all_con_penalty
        
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    
    return total_loss / len(loader.dataset)


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def MI_Est(discriminator, graph_embeddings, sub_embeddings):
    batch_size = graph_embeddings.shape[0]
    shuffle_embeddings = graph_embeddings[torch.randperm(batch_size)]
    
    joint = discriminator(graph_embeddings, sub_embeddings)
    margin = discriminator(shuffle_embeddings, sub_embeddings)
    
    # Donsker
    mi_est = torch.mean(joint) - torch.clamp(torch.log(torch.mean(torch.exp(margin))), -100000, 100000)
    
    return mi_est


def eval_acc(model, device, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred,_,_,_ = model(data)
            pred = pred.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, device, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out,_,_,_ = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)