
import torch
import torch.nn.functional as F


def vgib_train(model, classifier, optimizer, device, loader, args):
    model.train()
    
    total_loss = 0
    
    for data in loader: 
        data = data.to(device)
        num_graphs = len(data.y)
        
        embedding, noisy, pos_penalty, kl_loss, preserve_rate = model(data)
        features = noisy
        labels = data.y
        
        # features = torch.cat((embedding, noisy), dim = 0)
        # labels = torch.cat((data.y, data.y), dim = 0).to(device)
        
        pred = classifier(features)
        cls_loss = F.nll_loss(pred, labels)
        mi_loss = kl_loss
        
        optimizer.zero_grad()
        loss = cls_loss + args.con_weight * pos_penalty
        loss = loss + args.mi_weight * mi_loss
        loss.backward()
        
        total_loss += loss.item() * num_graphs
        optimizer.step()
    
    return total_loss / len(loader.dataset)


def vgib_eval(model, classifier, device, loader):
    model.eval()
    
    loss, correct = 0, 0
    for data in loader:
        data = data.to(device)
        
        with torch.no_grad():
            out, _, _, _, _ = model(data)
            out = classifier(out)
            pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    
    return loss / len(loader.dataset), correct / len(loader.dataset)
