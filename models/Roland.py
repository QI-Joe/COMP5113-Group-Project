from torch_geometric.nn import GCNConv, Linear
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell, CrossEntropyLoss
import sys
# sys.path.append("..")
from models.GraphSage import MeanAggregator, Encoder, SupervisedGraphSage, adjacent_list_building, node_splitation
import torch.nn as nn
from torch.nn import init
import argparse


def RoLand_config(model_detail):
    parser  = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="Cora")
    parser.add_argument("--hidden_conv1", type=int, default=64)
    parser.add_argument("--hidden_conv2", type=int, default=32)
    training_config = dict()
    parser=parser.parse_args(model_detail)
    for arg in vars(parser):
        training_config[arg] = getattr(parser, arg)
    
    return training_config

class ROLANDGNN(torch.nn.Module):
    def __init__(self, input_dim, num_nodes, device, dropout=0.0, update='moving'):
        
        super(ROLANDGNN, self).__init__()
        #Architecture: 
            #2 MLP layers to preprocess BERT repr, 
            #2 GCN layer to aggregate node embeddings
            #HadamardMLP as link prediction decoder
        
        #You can change the layer dimensions but 
        #if you change the architecture you need to change the forward method too
        #TODO: make the architecture parameterizable
        
        hidden_conv_1 = 64 
        hidden_conv_2 = 32
        self.preprocess1 = Linear(input_dim, 256).to(device)
        self.preprocess2 = Linear(256, 128).to(device)
        self.conv1 = GCNConv(128, hidden_conv_1).to(device)
        self.conv2 = GCNConv(hidden_conv_1, hidden_conv_2).to(device)
        self.postprocess1 = Linear(hidden_conv_2, 2).to(device)
        self.full_connect = Linear(input_dim, 128).to(device)

        self.encoder1, self.encoder2 = None, None
        self.encoder_ware = None
        #Initialize the loss function to BCEWithLogitsLoss
        # self.loss_fn = loss(reduction="mean")
        self.loss_fn = CrossEntropyLoss(reduction="mean")

        self.dropout = dropout
        self.update = update
        if update=='moving':
            self.tau = torch.Tensor([0]).to(device)
        elif update=='learnable':
            self.tau = torch.nn.Parameter(torch.Tensor([0])).to(device)
        elif update=='gru':
            self.gru1 = GRUCell(hidden_conv_1, hidden_conv_1).to(device)
            self.gru2 = GRUCell(hidden_conv_2, hidden_conv_2).to(device)
        elif update=='mlp':
            self.mlp1 = Linear(hidden_conv_1*2, hidden_conv_1).to(device)
            self.mlp2 = Linear(hidden_conv_2*2, hidden_conv_2).to(device)
        else:
            assert(update>=0 and update <=1)
            self.tau = torch.Tensor([update])
        self.previous_embeddings = [torch.zeros((num_nodes, hidden_conv_1)).cuda(), torch.zeros((num_nodes, hidden_conv_2)).cuda()]
        self.batch_embedding_cache: dict[int: torch.Tensor] = []
    
    def batch_emb_store(self, batch_idx, emb):
        self.batch_embedding_cache[batch_idx] = emb

    def batch_emb_matching(self, batch_idx):
        return self.batch_embedding_cache[batch_idx]

    def graph_weight(self, num_classes, embed_dim):
        self.sageweight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim).cuda())
        init.xavier_uniform_(self.sageweight)

    def GraphSage_feat_adjust(self, features: torch.Tensor, encoder: Encoder):
        new_feat = nn.Parameter(features, requires_grad=False).cuda()
        encoder.base_model.features = new_feat
        return encoder

    def GraphSage_warehouse(self, feature_shape: tuple[int], output_dim, adj_lists, device):
        """
        :param feature_shape: the shape of feature, (num_nodes, hidden_dim1|hidden_dim2)

        :return: a suitable encoder to able for 
        """
        num_nodes, feat_dim = feature_shape[0], feature_shape[1]*2
        features = nn.Embedding(num_nodes, feat_dim)
        features.weight = nn.Parameter(torch.FloatTensor(num_nodes, feat_dim).cuda(), requires_grad=False)
        init.xavier_uniform_(features.weight)

        agg1 = MeanAggregator(features, to_cuda=True)
        enc1 = Encoder(features, feat_dim, output_dim, adj_lists, agg1, gcn=True, to_cuda=True)
        agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), to_cuda=True)
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, output_dim, adj_lists, agg2,
                base_model=enc1, gcn=True, to_cuda=True)
        enc1.num_samples = 5
        enc2.num_samples = 5
        if self.encoder_ware is None:
            self.encoder_ware = [[agg1, agg2, enc1, enc2]]
        else:
            self.encoder_ware.append([agg1, agg2, enc1, enc2])
        
        if self.encoder1 == None:
            self.encoder1 = enc2
        elif self.encoder2 == None:
            self.encoder2 = enc2
        return enc2

    def Graphsage_encoder(self, node_fea, output_dim, adj_lists, device):
        """
        :param node_fea: node features

        :return: a suitable encoder to replease self.processing1
        """
        num_nodes, feat_dim = node_fea.shape[0], node_fea.shape[1]
        features = nn.Embedding(num_nodes, feat_dim) # (num_nodes, feature_dim)
        counter_weight =nn.Parameter(torch.empty([num_nodes, feat_dim], dtype=torch.float32, device="cuda"), requires_grad=False)
        with torch.no_grad():
            counter_weight.copy_(node_fea)
        features.weight = counter_weight

        agg1 = MeanAggregator(features, to_cuda=True)
        enc1 = Encoder(features, feat_dim, output_dim, adj_lists, agg1, gcn=True, to_cuda=True)
        agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), to_cuda=True)
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, output_dim, adj_lists, agg2,
                base_model=enc1, gcn=True, to_cuda=True)
        enc1.num_samples = 5
        enc2.num_samples = 5
        self.encoder = enc2
        self.encoder_keeper = [agg1, agg2, enc1, enc2]
        return self.encoder

    def reset_loss(self,loss=BCEWithLogitsLoss):
        self.loss_fn = loss()
        
    def reset_parameters(self):
        self.preprocess1.reset_parameters()
        self.preprocess2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocess1.reset_parameters()

    def forward(self, x, edge_index, graphsage: bool = False, gcn_only: bool=False, previous_embeddings=None, edge_label_index=None, num_current_edges=None, num_previous_edges=None):
        
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None: #None if test
            self.previous_embeddings = [previous_embeddings[0].clone(),previous_embeddings[1].clone()]
        """
        if self.update=='moving' and num_current_edges is not None and num_previous_edges is not None: #None if test
            #compute moving average parameter
            self.tau = torch.Tensor([num_previous_edges / (num_previous_edges + num_current_edges)]).clone() # tau -- past weight
        """
        current_embeddings = [torch.Tensor([]),torch.Tensor([])]
        
        #Preprocess text
        node_idx = [i for i in range(x.shape[0])]
        if graphsage and graphsage != 2:
            neighbor_emb = self.encoder(node_idx) # (embed_dim, num_nodes)
            # scores = self.sageweight.mm(neighbor_emb).t()
            x = neighbor_emb.t()
        # return scores, current_embeddings
        
        sum_dim1 = torch.sum(x, dim=1)
        if abs(torch.mean(sum_dim1).item()) >= 4.0:
            print("the position occur at GraphSage place, with value", sum_dim1[:5])

        # do some combination
        if not gcn_only:
            h = self.preprocess1(x)
            h = F.leaky_relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout,inplace=True)
            h = self.preprocess2(h)
            h = F.leaky_relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, inplace=True)
        else:
            h = self.full_connect(x)


        sum_dim1 = torch.sum(h, dim=1)
        if abs(torch.mean(sum_dim1).item()) >= 4.0:
            print("the position occur at MLP place, with value", sum_dim1[:5])

        #GRAPHCONV
        #GraphConv1
        h = self.conv1(h, edge_index)
        h = F.leaky_relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout,inplace=True) # 

        # normalization of past embedding
        prev_emb_norm = torch.norm(self.previous_embeddings[0], p=2, dim=1, keepdim=True)
        if not torch.sum(prev_emb_norm).item():
            prev_emb_norm = torch.ones(prev_emb_norm.shape).cuda()

        #Embedding Update after first layer, better h will also update here, not fucking simply detach and record
        """
        a aggregation operation on graphsage, should take action on... both side?
        """

        if self.update=='gru':
            h = self.gru1(h, self.previous_embeddings[0].clone() )  # / prev_emb_norm
        elif self.update=='mlp':
            hin = torch.cat((h,self.previous_embeddings[0].clone() / prev_emb_norm),dim=1) # /prev_emb_norm
            self.encoder1 = self.GraphSage_feat_adjust(hin.detach(), self.encoder)
            h = self.encoder1(node_idx).t()
        else:
            h = torch.Tensor(self.tau * self.previous_embeddings[0].clone() + (1-self.tau) * h.clone())
        
        h_norm = torch.norm(h, p=2, dim=1, keepdim=True)
        h = h/h_norm

        sum_dim1 = torch.sum(h, dim=1)
        if abs(torch.mean(sum_dim1).item()) >= 10.0:
            print("the position occur at Conv1 place, with value", sum_dim1[:5])

        current_embeddings[0] = h.detach()
        #GraphConv2
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout,inplace=True) # 
        
        # normalization of past embedding
        conv2_prev_norm = torch.norm(self.previous_embeddings[1], p=2, dim=1, keepdim=True)
        if not torch.sum(conv2_prev_norm).item():
            conv2_prev_norm = torch.ones(conv2_prev_norm.shape).cuda()

        #Embedding Update after second layer
        if self.update=='gru':
            h = self.gru2(h, self.previous_embeddings[1].clone() )  # / conv2_prev_norm
        elif self.update=='mlp':
            hin = torch.cat((h,self.previous_embeddings[1].clone() / conv2_prev_norm),dim=1) # /conv2_prev_norm
            # self.encoder2 = self.GraphSage_feat_adjust(hin.detach(), self.encoder2)
            # h = self.encoder2(node_idx).t()
            h = self.mlp2(hin)
        else:
            h = torch.Tensor(self.tau * self.previous_embeddings[1].clone() + (1-self.tau) * h.clone())
      
        # h_norm = torch.norm(h, p=2, dim=1, keepdim=True)
        # h = h/h_norm

        sum_dim1 = torch.sum(h, dim=1)
        if abs(torch.mean(sum_dim1).item()) >= 10.0:
            print("the position occur at Conv2 place, with value", sum_dim1[:5])

        current_embeddings[1] = h.detach()

        return h, current_embeddings
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
def RoLand_config(model_detail=None):
    parser  = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="Cora")
    parser.add_argument("--hidden_conv1", type=int, default=64)
    parser.add_argument("--hidden_conv2", type=int, default=32)
    training_config = dict()
    parser=parser.parse_args(model_detail)
    for arg in vars(parser):
        training_config[arg] = getattr(parser, arg)
    
    return training_config