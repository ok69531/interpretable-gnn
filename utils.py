import os
import random
import numpy as  np

import torch
import torch_geometric


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)


def save_model(file, path, name):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
    torch.save(file, os.path.join(path, name))
    print('Parameters are successfully saved.')
    