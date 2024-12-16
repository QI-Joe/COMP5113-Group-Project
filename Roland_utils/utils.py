import torch
from torch_geometric.data import Data
from typing import Union
import numpy as np
import random
# import matplotlib.pyplot as plt

def generate_negative_samples(data: Data, num_neg_samples: int = 1000):
    num_nodes = data.num_nodes
    edges = set(map(tuple, data.edge_index.t().tolist()))  # Existing edges
    negative_samples = []

    while len(negative_samples) < num_neg_samples:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if (src, dst) not in edges and (dst, src) not in edges and src != dst:
            negative_samples.append((src, dst))

    return torch.tensor(negative_samples, dtype=torch.long).t().contiguous()

def to_tensor(tensor: Union[Data, torch.Tensor]) -> torch.Tensor:
    if not isinstance(tensor.x, torch.Tensor):
        tensor.x = torch.tensor(tensor.x)
    if not isinstance(tensor.edge_index, torch.Tensor):
        tensor.edge_index = torch.from_numpy(tensor.edge_index)
    if not isinstance(tensor.y, torch.Tensor):
        tensor.y = torch.from_numpy(tensor.y)
    if hasattr(tensor, 'time') and not isinstance(tensor.time, torch.Tensor):
        tensor.time = torch.from_numpy(tensor.time)
    return tensor

def to_cuda(tensor: Union[Data|torch.Tensor], device: str) -> torch.Tensor:
    device = torch.device(device)

    if tensor.x.device != device:
        tensor.x = tensor.x.to(device)
    if tensor.edge_index.device != device:
        tensor.edge_index = tensor.edge_index.to(device)
    if tensor.y.device != device:
        tensor.y = tensor.y.to(device)
    
    pos_x_switch = False
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

def neg_pos_mix_perm(neg: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor]:
    pos_label = torch.ones(pos.shape[1])
    neg_label = torch.zeros(neg.shape[1])
    
    mixed = torch.cat((neg, pos), dim=1)
    mixed_label = torch.cat((neg_label, pos_label))

    perm = torch.randperm(mixed.shape[1])
    mixed = mixed[:, perm]
    mixed_label = mixed_label[perm]

    return mixed, mixed_label

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

# def plot_drawing(loss_data: list, epoch_num: int, ds_name): 
#     plt.figure(figsize=(10,5))
#     plt.plot(range(epoch_num), loss_data)
#     plt.title(f'{ds_name} Loss in epoch {epoch_num}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.xticks(range(0, epoch_num, 10))
#     plt.grid()
#     # save plot
#     plt.savefig(f'./{ds_name}_loss.png')
#     return