from models.Roland import ROLANDGNN, RoLand_config, adjacent_list_building
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import chain
import copy
import time
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from typing import Tuple, List
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch
from Roland_utils.dataloader import load_standard
from Roland_utils.utils import to_cuda, mask2index, index2mask
from Roland_utils.evaluator import LogRegression, Simple_Regression
import numpy as np
from Roland_utils.label_noise import label_process
import copy

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


def train_Roland(model: ROLANDGNN, projection_model: nn.Linear, train_data: Data,
                 num_classes: int, last_embeddings: list[torch.Tensor], optimizer, \
                 graphsage: bool = False, gcn_only: bool = False, \
                 num_current_edges=None, num_previous_edges=None,\
                 device='cpu', num_epochs=200, verbose=False) -> Tuple[ROLANDGNN, Optimizer, List[float], List[torch.Tensor], float]:
    avgpr_tra_max = 0
    model.to(device=device)
    projection_model.to(device)
    best_model: ROLANDGNN = model
    best_epoch=0
    val_regression: nn.Module = None
    best_current_embeddings = []
    f = F.log_softmax
    best_val, epoch_time = None, []
    
    tol = 5e-04
    loss_keper = 100
    
    current_embeddings=last_embeddings
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        projection_model.train()
        optimizer.zero_grad()

        # neighborloader = NeighborLoader(data=train_data, batch_size=1024, num_neighbors=[-1], shuffle=False)
        pred, current_embeddings =\
            model.forward(x=train_data.x, edge_index=train_data.edge_index, graphsage = graphsage, gcn_only = gcn_only, previous_embeddings=current_embeddings) 
        project_pred = f(projection_model(pred), dim=-1)
        # project_pred = pred
        loss = model.loss(project_pred[train_data.train_mask], train_data.y[train_data.train_mask]) 
        loss.backward() # retain_graph=True
        optimizer.step() 

        if abs(loss_keper - loss) >= 0.003:
            if loss_keper<100: print(f"loss changed from {loss_keper:03f} to {loss:03f} at epoch {epoch}")
            loss_keper = loss
            print(f"Epoch {epoch+1} | Loss {loss.item():.4f} ")

        epoch_time.append(time.time() - epoch_start)
        if (epoch+1) % 50 == 0: 
            model.eval()
            pred, _ = model.forward(x=train_data.x, edge_index=train_data.edge_index, graphsage = graphsage, \
                                    gcn_only = gcn_only, previous_embeddings=current_embeddings)
            val_metrics, val_regression = eval_Roland_SL(emb=pred, data=train_data, num_classes=num_classes, \
                                                         models=val_regression, is_val=True, is_test=False)
            if avgpr_tra_max-tol <= val_metrics["accuracy"]:
                avgpr_tra_max = val_metrics["accuracy"]
                best_epoch = epoch
                best_val = val_metrics
                best_current_embeddings = current_embeddings
                best_model = model
            print("val acc is {:.4f}".format(val_metrics["accuracy"]))
    
    if loss_keper!=100:
        print(f"final loss is {loss_keper:04f}, \n---------------------------------------------------- \n")

    
    return best_model, optimizer, best_val, best_current_embeddings, np.mean(epoch_time), val_regression


def main_Roland(configs, device='cpu'):
    # import dataset
    dataset_name = configs["dataset_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_snap = configs["snapshots"]
    hidden_conv1 = configs["hidden_conv1"]
    hidden_conv2 = configs["hidden_conv2"]
    noise_type = configs["noise_type"]
    noise_rate = configs["noise_rate"]
    seed = configs["seed"]
    graphsage = 2
    gcn_only = True

    graph = load_standard(dataset_name)[0]
    num_classes = graph.y.max().item()+1
    total_num_nodes = graph.x.shape[0]

    input_dim = graph.x.size(1) if not graphsage or graphsage==2 else hidden_conv1
    model = ROLANDGNN(input_dim, total_num_nodes, update='mlp', device=device)
    # graphsage_adj = adjacent_list_building(graph)
    model.reset_parameters()
    projection_model = LogRegression(hidden_conv2, num_classes=num_classes)

    optimizer = torch.optim.Adam(chain(model.parameters(), projection_model.parameters()), lr = 1e-2, weight_decay=1e-4)

    for t in range(num_snap-2):
        print(f"At Snapshot {t+1}, we observe loss as below, \n--------------------------------------------------")
        temporal_start = time.time()
        temporal_graph = graph
        temporal_adj = adjacent_list_building(temporal_graph)
        # x_counterpart = (copy.deepcopy(graph.pos), copy.deepcopy(graph.pos))
        # temporal_graph.pos = x_counterpart[0]
        num_nodes = temporal_graph.x.shape[0]
        
        last_embeddings = [torch.zeros((num_nodes, hidden_conv1)).to(device), \
                           torch.zeros((num_nodes, hidden_conv2)).to(device)]
        random.seed(2024)
        torch.manual_seed(2024)
        transform = RandomNodeSplit(num_val=0.1,num_test=0.8)

        # temporal_graph.establish_two_layer_idx_matching(idxloader)
        # transfered_graph = transfered_graph.mask_adjustment_two_layer_idx()
        label = copy.deepcopy(temporal_graph.y)
        label, modified_mask = label_process(label, num_classes, noise_type=noise_type,\
                                        noise_rate=noise_rate, random_seed=seed)
        temporal_graph.y = label
        transfered_graph: Data = transform(temporal_graph)
        transfered_graph = to_cuda(transfered_graph, device)

        # test_data = copy.deepcopy(temporal_dataloader.get_T1graph(t))
        # if dataset_name.lower() == "cora":
        #     test_data.test_mask = transfered_graph.test_mask
        # test_data.establish_two_layer_idx_matching(idxloader)
        # test_data.pos = x_counterpart[1]
        # test_data = to_cuda(test_data, device=device)

        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        if graphsage:
            model.graph_weight(num_classes=num_classes, embed_dim=hidden_conv1)
            model.Graphsage_encoder(node_fea=transfered_graph.x, output_dim=hidden_conv1, adj_lists=temporal_adj, device=device)

            model.GraphSage_warehouse((num_nodes, hidden_conv1), output_dim = hidden_conv1, adj_lists=temporal_adj, device=device)
            model.GraphSage_warehouse((num_nodes, hidden_conv1), output_dim = hidden_conv2, adj_lists=temporal_adj, device=device)

            optimizer = torch.optim.Adam(chain(model.parameters(), projection_model.parameters()), lr = 1e-2, weight_decay=1e-4)

        model, optimizer, stage_metrics, last_embeddings, avg_epoch, final_classifier =\
        train_Roland(model, projection_model, transfered_graph, num_classes, \
                     graphsage=graphsage, gcn_only=gcn_only, \
                    last_embeddings = last_embeddings, optimizer=optimizer, device=device)
        
        test_data = transfered_graph
        t1_emb, _ = model(test_data.x, test_data.edge_index, graphsage=graphsage, gcn_only=gcn_only, previous_embeddings=last_embeddings)
        m2, _ = eval_Roland_SL(emb=t1_emb, data=test_data, num_classes=num_classes, models=final_classifier, is_val=False, is_test=True, device=device)
        #SAVE AND DISPLAY EVALUATION
        m1 = stage_metrics
        print("Validation Final Trun: {}".format(round(m1["accuracy"], 4)))
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