from models.Roland import ROLANDGNN, RoLand_config, adjacent_list_building
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import chain
import time
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit
from typing import Tuple, List
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch
from Roland_utils.my_dataloader import data_load, Temporal_Dataloader, Temporal_Splitting, Dynamic_Dataloader
from Roland_utils.utils import to_cuda, to_tensor, generate_negative_samples, neg_pos_mix_perm
from Roland_utils.evaluator import LogRegression, Simple_Regression, LinkPredictor, link_evaluator
import numpy as np
from Roland_utils.label_noise import label_process
import torch.optim as optim
import re

def eval_Roland_SL(emb: torch.Tensor, data_pair: tuple[Data|torch.Tensor], models:nn.Linear, \
                   is_val: bool, is_test: bool, num_classes: int=2,\
                   device: str="cuda:0"):
    """
    in SL trianing that the validation and text is sperated not doing it together, thus the same learning MLP should be used
    data: needed due to we need correct label
    """
    # if is_val and not is_test:
    emb = emb.detach()
    return link_evaluator(emb, data_pairs=data_pair)
    # elif is_test and not is_val:
    #     emb = emb.detach()
    #     return link_evaluator(emb, data_pairs=data_pair)
    raise ValueError(f"is_val, is_test should not be the same. is_val: {is_val}, is_test: {is_test}")

def eval_model_Dy(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, device: str="cuda:0"):
    emb = emb.detach()
    truth = data.y.detach()
    return Simple_Regression(emb, truth, num_classes=num_classes, project_model=None, return_model=False)

def train_Roland(model: ROLANDGNN, projection_model: LinkPredictor, train_data: tuple[Temporal_Dataloader, Data], 
                 num_classes: int, last_embeddings: list[torch.Tensor], optimizer, \
                 val_pairs: tuple[Data, torch.Tensor], graphsage: bool = False, gcn_only: bool = False, \
                 device='cpu', num_epochs=200, verbose=False) -> Tuple[ROLANDGNN, optim.Optimizer, List[float], List[torch.Tensor], float]:
    avgpr_tra_max = 0
    val_data, val_neg_pair = val_pairs
    train_data, train_neg_pair = train_data
    # model.to(device=device)
    projection_model.to(device)
    best_model: ROLANDGNN = model
    best_epoch=0
    val_regression: nn.Module = None
    best_current_embeddings = []
    f = F.log_softmax
    best_val, epoch_time = None, []
    
    train_pair = train_data.edge_index[:, train_data.edge_label]
    train, train_label = neg_pos_mix_perm(neg=train_neg_pair, pos=train_pair)

    pos_pair = val_data.edge_index[:, val_data.edge_label]
    val, val_label = neg_pos_mix_perm(neg=val_neg_pair, pos = pos_pair)

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
        z_src = pred[train[0,:]]
        z_dst = pred[train[1,:]]
        
        project_pred = projection_model.forward(z_src=z_src, z_dst=z_dst)

        loss = model.loss(project_pred, train_label.to(device)) 
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

    graph = data_load(dataset_name)
    num_classes = 2 # graph.y.max().item()+1
    input_dim = 64
    temporal_dataloader = graph

    if re.search(r'\bbit\b.*\balpha\b|\balpha\b.*\bbit\b', dataset_name, flags=re.IGNORECASE):
        temporal_list = Temporal_Splitting(graph).temporal_splitting(time_mode="view", \
                        snapshot=snapshot, views=snapshot-2, strategy="sequential", non_split=non_split)
        temporal_dataloader = Dynamic_Dataloader(temporal_list, graph)
        input_dim = graph.pos.size(1)
    
    mlp_hidden = (512, 256)
    conv_hidden = (hidden_conv1, hidden_conv2)

    model = ROLANDGNN(input_dim, mlp_hidd=mlp_hidden, conv_hidd=conv_hidden, update='mlp', device=device)

    model.reset_parameters()
    projection_model = LinkPredictor(in_channels=hidden_conv2) # LogRegression(hidden_conv2, num_classes=num_classes)

    optimizer = torch.optim.Adam(chain(model.parameters(), projection_model.parameters()), lr = 1e-2, weight_decay=1e-4)

    for t in range(snapshot-2):
        print(f"At Snapshot {t+1}, we observe loss as below, \n--------------------------------------------------")
        temporal_start = time.time()
        temporal_graph = temporal_dataloader.get_temporal()
        num_links = temporal_graph.num_edges
        
        last_embeddings = [torch.zeros((num_links, hidden_conv1)).to(device), \
                           torch.zeros((num_links, hidden_conv2)).to(device)]
        random.seed(2024)
        torch.manual_seed(2024)
        transform = RandomLinkSplit(is_undirected=True, num_val=0.1,num_test=0.0)
        temporal_graph = to_tensor(temporal_graph)

        temp_data = Data(x=temporal_graph.x, edge_index=temporal_graph.edge_index)
        train_data, val_data, test_data = transform.forward(temp_data)

        train_negative_pair = generate_negative_samples(temporal_graph)
        val_negative_pair = generate_negative_samples(temporal_graph)

        model, optimizer, stage_metrics, last_embeddings, avg_epoch, final_classifier =\
        train_Roland(model, projection_model, num_classes, \
                     train_data=(train_data, train_negative_pair), val_data=(val_data, val_negative_pair), \
                     graphsage=graphsage, gcn_only=gcn_only, \
                    last_embeddings = last_embeddings, optimizer=optimizer, device=device)
        
        t1_temporal = temporal_dataloader.get_T1graph(t)
        negative_pair = generate_negative_samples(temporal_graph)
        test_data: Temporal_Dataloader = to_cuda(t1_temporal)
        with torch.no_grad():
            t1_emb, _ = model.forward(test_data.x, test_data.edge_index, graphsage=graphsage, \
                              gcn_only=gcn_only, previous_embeddings=None)
            
        m2, _ = eval_model_Dy(emb=t1_emb, data=test_data, num_classes=num_classes)
        #SAVE AND DISPLAY EVALUATION
        m1 = stage_metrics

        model.reset_parameters()
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