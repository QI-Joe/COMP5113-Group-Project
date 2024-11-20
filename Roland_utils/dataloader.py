import torch_geometric as pyg
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import os.path as osp
from torch_geometric.datasets import Coauthor, WikiCS, Amazon, Planetoid, CitationFull

def get_dataset(path, name: str):
    assert name.lower() in [val.lower() for val in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']]
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    # if name.startswith('ogbn'):
    #     return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())

def load_standard(dataset: str, *wargs) -> tuple[Data]:
    
    path = osp.expanduser('~/datasets')
    path = osp.join(path, dataset)
    dataset = get_dataset(path, dataset)
    return dataset


class Dataloader():
    def __init__(self):
        pass