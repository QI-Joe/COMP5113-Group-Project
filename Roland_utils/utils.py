import torch
from torch_geometric.data import Data
from typing import Union
import numpy as np
import random

def to_cuda(tensor: Union[Data|torch.Tensor], device: str) -> torch.Tensor:
    device = torch.device(device)
    if tensor.x.device != device:
        tensor.x = tensor.x.to(device)
    if tensor.edge_index.device != device:
        tensor.edge_index = tensor.edge_index.to(device)
    if tensor.y.device != device:
        tensor.y = tensor.y.to(device)
    
    pos_x_switch = True
    try:
        tensor.pos = tensor.pos.to(device)
    except Exception as e:
        pos_x_switch = False
        pass
    if pos_x_switch:
        temp = tensor.pos
        tensor.x = tensor.pos
        tensor.pos = temp

    return tensor

def setup_seed(seed):
    '''
    Setup random seed so that the experimental results are reproducible
    Parameters
    ----------
    seed : int
        random seed for torch, numpy and random

    Returns
    -------
    None
    '''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mask2index(mask: torch.Tensor) -> torch.Tensor:
    '''
    Convert mask to index
    '''
    return torch.arange(mask.size(0))[mask]

def index2mask(index: torch.Tensor, size: int) -> torch.Tensor:
    '''
    Convert index to mask
    '''
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask