import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_sparse import transpose

from torch_geometric.utils import sort_edge_index, is_undirected
from torch_geometric.nn import GINConv, global_add_pool, InstanceNorm

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class GIN(nn.Module):
    def __init__(self, input_dim, num_class, args):
        super(GIN, self).__init__()

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool
        
        self.hidden_size = args.hidden_size
        self.dropout_p = args.dropout_p
        self.n_layers = args.n_layers
        self.multi_label = args.multi_label

        self.convs.append(
        GINConv(
            nn.Sequential(
                nn.Linear(input_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size)
            ), train_eps = False
        )
        )

        for i in range(self.n_layers - 1):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.hidden_size)
                    ), train_eps = False
                )
            )

        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size, 1 if num_class == 2 and not self.multi_label else num_class)
        )
    
    def forward(self, data):
        h = self.convs[0](data.x, data.edge_index)
        h = self.relu(h)
        h = F.dropout(h, p = self.dropout_p, training = self.training)
        
        for conv in self.convs[1:]:
            h = conv(h, data.edge_index)
            h = self.relu(h)
            h = F.dropout(h, p = self.dropout_p, training = self.training)
        
        return self.fc_out(self.pool(h, data.batch))
    
    def get_emb(self, data):
        h = self.convs[0](data.x, data.edge_index)
        h = self.relu(h)
        h = F.dropout(h, p = self.dropout_p, training = self.training)
        
        for conv in self.convs[1:]:
            h = conv(h, data.edge_index)
            h = self.relu(h)
            h = F.dropout(h, p = self.dropout_p, training = self.training)
        
        return h
    
    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias = True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i-1], channels[i], bias))
            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))
        super(MLP, self).__init__(*m)


class Criterion(nn.Module):
    def __init__(self, num_class, multi_label):
        super(Criterion, self).__init__()
        
        self.num_class = num_class
        self.multi_label = multi_label
        print(f'[INFO] Using multi_label: {self.multi_label}')
    
    def forward(self, logits, targets):
        if self.num_class == 2 and not self.multi_label:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        elif self.num_class > 2 and not self.multi_label:
            loss = F.cross_entropy(logits, targets.long())
        else:
            is_labeled = targets == targets
            loss = F.binary_cross_entropy_with_logits(logits[is_labeled], targets[is_labeled].float())
        return loss


class ExtractorMLP(nn.Module):
    def __init__(self, shared_config):
        super(ExtractorMLP, self).__init__()
        
        self.hidden_size = shared_config.hidden_size
        self.learn_edge_att = shared_config.learn_edge_att
        self.dropout_p = shared_config.extractor_dropout_p
        
        if self.learn_edge_att:
            self.feature_extractor = MLP([self.hidden_size * 2, self.hidden_size * 4, self.hidden_size, 1],
                                         dropout = self.dropout_p)
        else:
            self.feature_extractor = MLP([self.hidden_size * 1, self.hidden_size * 2, self.hidden_size, 1],
                                         dropout = self.dropout_p)
        
    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim = -1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        
        return att_log_logits


def reorder_like(from_edge_index, to_edge_index, values):
    from_edge_index, values = sort_edge_index(from_edge_index, values)
    ranking_score = to_edge_index[0] * (to_edge_index.max()+1) + to_edge_index[1]
    ranking = ranking_score.argsort().argsort()
    if not (from_edge_index[:, ranking] == to_edge_index).all():
        raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
    return values[ranking]


def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float()
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds


def process_data(data, use_edge_attr):
    if not use_edge_attr:
        data.edge_attr = None
    if data.get('edge_label', None) is None:
        data.edge_label = torch.zeros(data.edge_index.shape[1])
    return data


class GSAT(nn.Module):
    def __init__(self, clf, extractor, optimizer, device, num_class, args):
        super().__init__()
        
        self.clf = clf
        self.extractor = extractor
        self.optimizer = optimizer
        self.num_class = num_class
        
        self.criterion = Criterion(num_class, args.multi_label)
        self.args = args
        self.device = device
    
    def __loss__(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)
        
        r = self.args.fix_r if self.args.fix_r else self.get_r(self.args.decay_interval, self.args.decay_r, epoch, self.args.final_r, self.args.init_r)
        info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1-r+1e-6) + 1e-6)).mean()
        
        pred_loss = pred_loss * self.args.pred_loss_coef
        info_loss = info_loss * self.args.info_loss_coef
        loss = pred_loss + info_loss
        loss_dict = {
            'loss': loss.item(),
            'pred': pred_loss.item(),
            'info': info_loss.item()
        }
        
        return loss, loss_dict
    
    def forward_pass(self, data, epoch, training):
        emb = self.clf.get_emb(data)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, training)

        if self.args.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced = False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        clf_logits = self.clf(data)
        if self.num_class > 2:
            loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch)
        else:
            loss, loss_dict = self.__loss__(att, clf_logits, data.y.view(clf_logits.shape), epoch)

        return edge_att, loss, loss_dict, clf_logits
    
    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
    
    def sampling(self, att_log_logits, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att
    
    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att
    
    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
