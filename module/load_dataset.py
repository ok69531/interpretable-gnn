import re
import os
import pickle
import numpy as np
from tqdm import tqdm

import rdkit.Chem as Chem

import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import TUDataset, MNISTSuperpixels

from .feature_expansion import FeatureExpander


def get_dataset(root, dataset_name):
    tud_names = ['MUTAG', 'Mutagenicity', 'PROTEINS', 'DD', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI']
    interpret_names = ['QED', 'DRD2', 'HLM', 'RLM', 'MLM']
    syn_names = ['BA-2motif']
    
    if dataset_name in tud_names:
        if 'IMDB' in dataset_name:
            pre_transform = FeatureExpander().transform
            return TUDataset(root, dataset_name, pre_transform = pre_transform)
        else:
            return TUDataset(root, dataset_name)
    elif dataset_name in interpret_names:
        return InterpretDataset(root, dataset_name)
    elif dataset_name in syn_names:
        return BA2MotifDataset(root, dataset_name)
    elif dataset_name == 'MNIST':
        return MNISTSuperpixels(os.path.join(root, 'MNIST'), train = False)


# ----------------------- Interpretation Data ----------------------- #
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


def preprocess_interpret_data(mol, pairs):
    smiles = pairs[0]
    props = float(pairs[1])
    processed = {}
    processed['label'] = props
    processed['smiles'] = smiles

    edge = []
    for bond in mol.GetBonds():
        edge.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    processed['edges'] = edge

    node = {}
    for atom in mol.GetAtoms():
        node[str(atom.GetIdx())] = atom.GetSymbol()
    processed['features'] = node
    
    return processed


def read_interpret_data(root, dataset_name):
    path = os.path.join(root, dataset_name, 'raw', dataset_name.lower()+'.txt')
    
    with open(path, 'r') as f:
        tmp = f.readlines()
    data = [i.split(' ') for i in tmp]

    processed_data = []
    
    for pairs in data:
        mol = get_mol(pairs[0])

        if mol is not None:
            processed = preprocess_interpret_data(mol, pairs)
            processed_data.append(processed)
    
    return processed_data


class InterpretDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, transform = None, pre_transform = None, empty = False):
        self.root = root
        self.dataset_name = dataset_name.upper()
        
        if self.dataset_name not in ['QED', 'DRD2', 'HLM', 'RLM', 'MLM']:
            raise ValueError(f'Invalid dataset: {self.dataset_name}')
        
        super(InterpretDataset, self).__init__(root, transform, pre_transform)
        
        if not empty:
            if not os.path.exists(self.processed_paths[0]):
                self.preprocess()
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    def _enumerate_graphs(self):
        self.graphs = read_interpret_data(self.root, self.dataset_name)
        self.graph_count = len(self.graphs)

        labels = set()
        features = set()
        for i in tqdm(range(self.graph_count)):
            data = self.graphs[i]
            labels = labels.union(set([data['label']]))
            features = features.union(set([val for k, v in data['features'].items() for val in v]))
        self.label_map = {v: i for i, v in enumerate(labels)}
        self.feature_map = {v: i for i, v in enumerate(features)}

    def _count_features_and_labels(self):
        self.number_of_features = len(self.feature_map)
        self.number_of_labels = len(self.label_map)

    def _create_target(self):
        target = [g['label'] for g in self.graphs]

    def _transform_edges(self, raw_data):
        edges = [[edge[0], edge[1]] for edge in raw_data['edges']]
        edges = edges + [[edge[1], edge[0]] for edge in raw_data['edges']]
        return torch.t(torch.LongTensor(edges))

    def _transform_features(self, raw_data):
        number_of_nodes = len(raw_data['features'])
        feature_matrix = np.zeros((number_of_nodes, self.number_of_features))
        index1 = [int(n) for n, feats in raw_data['features'].items() for f in feats]
        index2 = [int(self.feature_map[f]) for n, feats in raw_data['features'].items() for f in feats]
        feature_matrix[index1, index2] = 1.
        return torch.FloatTensor(feature_matrix)

    def _data_transform(self, raw_data):
        clean_data = Data(x = self._transform_features(raw_data),
                          edge_index = self._transform_edges(raw_data),
                          y = raw_data['label'],
                          smmiles = raw_data['smiles'])
        return clean_data

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.dataset_name, 'raw')
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dataset_name, 'processed')
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def preprocess(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        self._enumerate_graphs()
        self._count_features_and_labels()
        self._create_target()
        
        data_list = [self._data_transform(graph) for graph in self.graphs]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# ---------------------------- BA-2Motifs ---------------------------- #
def read_ba2motif_data(root, dataset_name = 'BA-2motif'):
    with open(os.path.join(root, dataset_name, 'raw', dataset_name + '.pkl'), 'rb') as f:
        adjs, features, labels = pickle.load(f)
    
    data_list = []
    for i in range(adjs.shape[0]):
        data_list.append(
            Data(x = torch.from_numpy(features[i]).float(),
                 edge_index = dense_to_sparse(torch.from_numpy(adjs[i]))[0],
                 y = torch.from_numpy(np.where(labels[i])[0]))
        )
    
    return data_list


class BA2MotifDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, transform = None, pre_transform = None):
        
        self.root = root
        self.dataset_name = dataset_name
        super(BA2MotifDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, self.dataset_name, 'raw')
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dataset_name, 'processed')
    
    @property
    def raw_file_names(self):
        return [f'{self.dataset_name}.pkl']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data_list = read_ba2motif_data(self.root, self.dataset_name)
        
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
        
        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == '__main__':
    root = 'dataset'
    
    dataset = TUDataset(root, 'MUTAG')
    print(dataset[0])
    
    dataset = InterpretDataset(root, 'QED')
    print(dataset[0])
    
    dataset = BA2MotifDataset(root, 'BA-2motif')
    print(dataset[0])

