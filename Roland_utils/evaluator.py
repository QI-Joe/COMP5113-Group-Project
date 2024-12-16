import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

class LogRegression(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LogRegression, self).__init__()
        self.lin = torch.nn.Linear(in_channels, num_classes)
        nn.init.xavier_uniform_(self.lin.weight.data)
        # torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.lin(x)
        return ret

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = nn.Linear(in_channels, in_channels)
        self.lin_dst = nn.Linear(in_channels, in_channels)
        nn.init.xavier_normal_(self.lin_src.weight.data)
        nn.init.xavier_normal_(self.lin_dst.weight.data)
        self.lin_final = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src)+self.lin_dst(z_dst)
        h = h.relu()
        # h = F.cosine_similarity(self.lin_src(z_src), self.lin_dst(z_dst))
        return torch.sigmoid(self.lin_final(h))

def link_evaluator(embedding: torch.Tensor, data_pairs: tuple[torch.Tensor], num_classes: int=2, num_epochs: int = 1500):
    device = embedding.device

    eval_data, eval_pair_label = data_pairs

    evaluator_model = LinkPredictor(in_channels=embedding.shape[1]).to(device)
    optimizer = Adam(evaluator_model.parameters(), lr=0.01, weight_decay=1e-4)

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        evaluator_model.train()
        optimizer.zero_grad()

        z_src = embedding[eval_data[0, :]]
        z_dst = embedding[eval_data[1, :]]

        output = evaluator_model.forward(z_src=z_src, z_dst=z_dst)
        loss = loss_fn(output.view(-1), eval_pair_label.to(device))

        loss.backward(retain_graph=False)
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f'LogRegression | Epoch {epoch}: loss {loss.item():.4f}')
    
    with torch.no_grad():
        eval_emb = evaluator_model.forward(z_src=z_src, z_dst=z_dst).view(-1)
        projection = (eval_emb>0.5).int()
        y_true, y_hat = eval_pair_label.cpu().numpy(), projection.cpu().numpy()
        accuracy, precision, recall, f1 = accuracy_score(y_true, y_hat), \
                                        precision_score(y_true, y_hat, average='macro', zero_division=0), \
                                        recall_score(y_true, y_hat, average='macro'),\
                                        f1_score(y_true, y_hat, average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def mean_reciprocal_rank(src: torch.Tensor, dst: torch.Tensor, truth: np.ndarray):
    """
    Calculate Mean Reciprocal Rank (MRR) for link prediction.

    Args:
        predictions (torch.Tensor): Predicted scores for all possible edges.
        positive_edge_index (torch.Tensor): Positive edges (shape: [2, P]) where P is the number of positive edges.

    Returns:
        float: Mean Reciprocal Rank score.
    """
    ranks = []
    dst= dst.numpy()
    
    length = src.size(0)
    for i in range(length):
        # Create a list of edges with their scores
        target = src[i].numpy() # 1x7
        target_norm = np.linalg.norm(target)
        target_normalized = target/target_norm

        matrix_norm = np.linalg.norm(dst) # nx7
        dst_normalized = dst/matrix_norm

        cos_sim = np.dot(dst_normalized, target_normalized.T) # nx1
        cos_index = np.argsort(cos_sim)
        cos_sim = cos_sim[cos_index]

        truth = truth[cos_index]

        src_true_dst = truth[i]

        # Find the position of src_true_dst in sorted_score
        position = np.where(truth == src_true_dst)[0][0]
        ranks.append(position + 1)  # Rank is 1-based
        # array[array[:, 0].argsort()]

    # Calculate MRR
    mrr = sum(1.0 / rank for rank in ranks) / len(ranks)
    return mrr

def Simple_Regression(embedding: torch.Tensor, label: Union[torch.Tensor | np.ndarray], num_classes: int, \
                      num_epochs: int = 1500,  project_model=None, return_model: bool = False) -> tuple[float, float, float, float]:
    device = embedding.device
    if not isinstance(label, torch.Tensor):
        label = torch.LongTensor(label).to(device)
    linear_regression = LogRegression(embedding.size(1), num_classes).to(device) if project_model==None else project_model
    f = nn.LogSoftmax(dim=-1)
    optimizer = Adam(linear_regression.parameters(), lr=0.01, weight_decay=1e-4)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        linear_regression.train()
        optimizer.zero_grad()
        output = linear_regression(embedding)
        loss = loss_fn(f(output), label)

        loss.backward(retain_graph=False)
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f'LogRegression | Epoch {epoch}: loss {loss.item():.4f}')

    with torch.no_grad():
        projection = linear_regression(embedding)
        y_true, y_hat = label.cpu().numpy(), projection.argmax(-1).cpu().numpy()
        accuracy, precision, recall, f1 = accuracy_score(y_true, y_hat), \
                                        precision_score(y_true, y_hat, average='macro', zero_division=0), \
                                        recall_score(y_true, y_hat, average='macro'),\
                                        f1_score(y_true, y_hat, average='macro')
        prec_micro, recall_micro, f1_micro = precision_score(y_true, y_hat, average='micro', zero_division=0), \
                                            recall_score(y_true, y_hat, average='micro'),\
                                            f1_score(y_true, y_hat, average='micro')
    if return_model:
        return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, linear_regression
    
    return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, None
