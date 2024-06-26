# https://github.com/Samyu0304/graph-information-bottleneck-for-Subgraph-Recognition/tree/main

import torch
import torch.nn.functional as F


def gib_train(model, discriminator, optimizer, local_optimizer, device, loader, args):
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


def gib_eval(model, device, loader):
    model.eval()
    
    loss, correct = 0, 0
    for data in loader:
        data = data.to(device)
        
        with torch.no_grad():
            out, _, _, _ = model(data)
            pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    
    return loss / len(loader.dataset), correct / len(loader.dataset)
