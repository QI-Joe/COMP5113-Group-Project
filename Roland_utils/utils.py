import torch
from torch_geometric.data import Data
from typing import Union

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