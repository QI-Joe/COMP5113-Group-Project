from models.Roland import ROLANDGNN, RoLand_config, adjacent_list_building
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import chain
import time
from torch_geometric.transforms import RandomNodeSplit
from typing import Tuple, List
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch
from Roland_utils.my_dataloader import data_load, Temporal_Dataloader, Temporal_Splitting, Dynamic_Dataloader
from Roland_utils.utils import to_cuda
from Roland_utils.evaluator import LogRegression, Simple_Regression
import numpy as np
from Roland_utils.label_noise import label_process
import torch.optim as optim

def eval_Roland_SL(emb: torch.Tensor, data: Data, num_classes: int, models:nn.Linear, \
                   is_val: bool, is_test: bool, \
                   device: str="cuda:0", split_ratio: float=0.1):
    """
    in SL trianing that the validation and text is sperated not doing it together, thus the same learning MLP should be used
    data: needed due to we need correct label
    """
    if is_val and not is_test:
        emb = emb[data.val_mask].detach()
        truth = data.y[data.val_mask].detach()
        return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=True, num_epochs=2000)
    elif is_test and not is_val:
        # ground_node_mask = data.layer2_n_id.index_Temporal.values
        test_indices = data.test_mask
        emb = emb[test_indices].detach()
        truth = data.y[test_indices].detach()
        return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=False, num_epochs=2000)
    raise ValueError(f"is_val, is_test should not be the same. is_val: {is_val}, is_test: {is_test}")

def eval_model_Dy(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, device: str="cuda:0"):
    emb = emb.detach()
    truth = data.y.detach()
    return Simple_Regression(emb, truth, num_classes=num_classes, project_model=None, return_model=False)

def train_Roland(model: ROLANDGNN, projection_model: nn.Linear, train_data: Temporal_Dataloader, 
                 num_classes: int, last_embeddings: list[torch.Tensor], optimizer, \
                 graphsage: bool = False, gcn_only: bool = False, \
                 device='cpu', num_epochs=200, verbose=False) -> Tuple[ROLANDGNN, optim.Optimizer, List[float], List[torch.Tensor], float]:
    avgpr_tra_max = 0
    # model.to(device=device)
    projection_model.to(device)
    best_model: ROLANDGNN = model
    best_epoch=0
    val_regression: nn.Module = None
    best_current_embeddings = []
    f = F.log_softmax
    best_val, epoch_time = None, []
    
    tol = 5e-04
    loss_keper = 100
    MRR, Past_MRR, threshold, epoch = 1, 0, 1e-2, 0
    
    current_embeddings=last_embeddings
    while MRR - Past_MRR > threshold and epoch < num_epochs:
        epoch_start = time.time()
        model.train()
        projection_model.train()
        optimizer.zero_grad()

        # neighborloader = NeighborLoader(data=train_data, batch_size=1024, num_neighbors=[-1], shuffle=False)
        pred, current_embeddings =\
            model.forward(x=train_data.x, edge_index=train_data.edge_index, graphsage = graphsage, gcn_only = gcn_only, previous_embeddings=current_embeddings) 
        project_pred = f(projection_model(pred), dim=-1)
        loss = model.loss(project_pred[train_data.train_mask], train_data.y[train_data.train_mask]) 
        loss.backward() # retain_graph=True
        optimizer.step() 

        if abs(loss_keper - loss) >= 3e-3:
            if loss_keper<100: print(f"loss changed from {loss_keper:03f} to {loss:03f} at epoch {epoch}")
            loss_keper = loss
            print(f"Epoch {epoch+1} | Loss {loss.item():.4f} ")

        epoch_time.append(time.time() - epoch_start)
        if (epoch+1) % 50 == 0: 
            model.eval()
            pred, _ = model.forward(x=train_data.x, edge_index=train_data.edge_index, graphsage = graphsage, gcn_only = gcn_only, previous_embeddings=current_embeddings)
            val_metrics, val_regression = eval_Roland_SL(emb=pred, data=train_data, num_classes=num_classes, models=val_regression, is_val=True, is_test=False)
            if avgpr_tra_max-tol <= val_metrics["accuracy"]:
                avgpr_tra_max = val_metrics["accuracy"]
                best_epoch = epoch
                best_val = val_metrics
                best_current_embeddings = current_embeddings
                best_model = model
            print("val acc is {:.4f}".format(val_metrics["accuracy"]))
        epoch += 1
        
    if loss_keper!=100:
        print(f"final loss is {loss_keper:04f}, \n---------------------------------------------------- \n")

    
    return best_model, optimizer, best_val, best_current_embeddings, np.mean(epoch_time), val_regression


def main_Roland(configs, device='cpu'):
    r"""
    RoLAND is expected to cancel the midset of GraphSage and only use 2 GCN with softten mixed MLP
    """
    # import dataset
    dataset_name = configs["dataset_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot = configs["snapshots"]
    hidden_conv1 = configs["hidden_conv1"]
    hidden_conv2 = configs["hidden_conv2"]
    graphsage = False
    gcn_only = True
    non_split = True

    graph, idxloader = data_load(dataset_name)
    num_classes = graph.y.max().item()+1
    temporal_list = Temporal_Splitting(graph).temporal_splitting(time_mode="view", \
                    snapshot=snapshot, views=snapshot-2, strategy="sequential", non_split=non_split)
    temporal_dataloader = Dynamic_Dataloader(temporal_list, graph)
    total_num_nodes = graph.x.shape[0]

    input_dim = graph.pos.size(1) if not graphsage or graphsage==2 else hidden_conv1
    mlp_hidden = (512, 256)
    conv_hidden = (hidden_conv1, hidden_conv2)

    model = ROLANDGNN(input_dim, total_num_nodes, mlp_hidd=mlp_hidden, conv_hidd=conv_hidden, update='mlp', device=device)

    model.reset_parameters()
    projection_model = LogRegression(hidden_conv2, num_classes=num_classes)

    optimizer = torch.optim.Adam(chain(model.parameters(), projection_model.parameters()), lr = 1e-2, weight_decay=1e-4)

    for t in range(snapshot-2):
        print(f"At Snapshot {t+1}, we observe loss as below, \n--------------------------------------------------")
        temporal_start = time.time()
        temporal_graph = temporal_dataloader.get_temporal()
        num_nodes = temporal_graph.pos.shape[0]
        
        last_embeddings = [torch.zeros((num_nodes, hidden_conv1)).to(device), \
                           torch.zeros((num_nodes, hidden_conv2)).to(device)]
        random.seed(2024)
        torch.manual_seed(2024)
        transform = RandomNodeSplit(num_val=0.2,num_test=0.0)

        transfered_graph: Temporal_Dataloader = transform(temporal_graph)
        transfered_graph = to_cuda(transfered_graph, device)

        t1_temporal = temporal_dataloader.get_T1graph(t)

        model, optimizer, stage_metrics, last_embeddings, avg_epoch, final_classifier =\
        train_Roland(model, projection_model, transfered_graph, num_classes, \
                     graphsage=graphsage, gcn_only=gcn_only, \
                    last_embeddings = last_embeddings, optimizer=optimizer, device=device)
        
        test_data: Temporal_Dataloader = to_cuda(t1_temporal)
        with torch.no_grad():
            t1_emb, _ = model.forward(test_data.x, test_data.edge_index, graphsage=graphsage, \
                              gcn_only=gcn_only, previous_embeddings=None)
            
        m2, _ = eval_model_Dy(emb=t1_emb, data=test_data, num_classes=num_classes)
        #SAVE AND DISPLAY EVALUATION
        m1 = stage_metrics

        print("Validation Final Trun: {}".format(round(m1["accuracy"], 4)))
        temporal_dataloader.update_event(t)
        temporal_time = time.time() - temporal_start
        print(f'View {t+1}, \n \
                Average Epoch Time {avg_epoch}, \n \
                Temporal Time {temporal_time}, \n \
                Val Acc {m1["accuracy"]:05f}, \n \
                Test Acc {m2["accuracy"]:05f}, \n \
                Avg Test precision {m2["precision"]:05f}, \n \
                Avg Test recall {m2["recall"]:05f}, \n \
                Avg Test f1 {m2["f1"]:05f}')
        
if __name__ == "__main__":
    configs = RoLand_config()
    main_Roland(configs, device='cpu')