import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def process_data(data, use_edge_attr):
    if not use_edge_attr:
        data.edge_attr = None
    if data.get('edge_label', None) is None:
        data.edge_label = torch.zeros(data.edge_index.shape[1])
    return data


@torch.no_grad()
def eval_one_batch(model, data, epoch):
    model.clf.eval()
    model.extractor.eval()

    att, loss, loss_dict, clf_logits = model.forward_pass(data, epoch, training = False)

    return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()    


def train_one_batch(model, data, epoch):
    model.extractor.train()
    model.clf.train()

    att, loss, loss_dict, clf_logits = model.forward_pass(data, epoch, training=True)
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()


def run_one_epoch(model, data_loader, epoch, phase, device, gsat_args):
    loader_len = len(data_loader)
    
    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    
    all_exp_labels, all_att, all_clf_labels, all_clf_logits = ([] for i  in range(4))
    
    for idx, data in enumerate(data_loader):
        data = process_data(data, gsat_args.use_edge_attr)
        att, loss_dict, clf_logits = run_one_batch(model, data.to(device), epoch)
        
        all_loss_dict = {}
        exp_labels = data.edge_label.data.cpu()
        
        for k, v in loss_dict.items():
            all_loss_dict[k] = all_loss_dict.get(k, 0) + v
        
        all_exp_labels.append(exp_labels), all_att.append(att)
        all_clf_labels.append(data.y.data.cpu().view(len(data.y), -1)), all_clf_logits.append(clf_logits)

        if idx == loader_len - 1:
            all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
            all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

            for k, v in all_loss_dict.items():
                all_loss_dict[k] = v / loader_len
            clf_acc, clf_roc, att_auroc = get_eval_score(all_exp_labels, all_att, all_clf_labels, all_clf_logits, gsat_args.multi_label)
            
    return all_loss_dict['loss'], clf_acc, clf_roc, att_auroc 


def get_eval_score(exp_labels, att, clf_labels, clf_logits, multi_label):
    clf_preds = get_preds(clf_logits, multi_label)
    clf_acc = 0 if multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

    clf_roc = 0
    # if 'ogb' in dataset_name:
    #     evaluator = Evaluator(name='-'.join(dataset_name.split('_')))
    #     clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']
    att_auroc = roc_auc_score(exp_labels, att) if np.unique(exp_labels).shape[0] > 1 else 0

    return clf_acc, clf_roc, att_auroc 


def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float().view(len(logits), -1)
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds
