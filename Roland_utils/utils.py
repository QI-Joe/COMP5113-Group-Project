import torch
from torch_geometric.data import Data
from typing import Union

def to_cuda(tensor: Union[Data|torch.Tensor], device: str) -> torch.Tensor:
    device = torch.device(device)
    if not isinstance(tensor.x, torch.Tensor):
        tensor.x = torch.tensor(tensor.x).to(device)
    if not isinstance(tensor.edge_index, torch.Tensor):
        tensor.edge_index = torch.tensor(tensor.edge_index).to(device)
    if not isinstance(tensor.edge_attr, torch.Tensor):
        tensor.edge_attr = torch.tensor(tensor.edge_attr).to(device)
    if not isinstance(tensor.y, torch.Tensor) or tensor.y.device != device:
        tensor.y = torch.tensor(tensor.y).to(device)
    
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