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


def RoLand_config(model_detail=None):
    parser  = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="autos")
    parser.add_argument("--hidden_conv1", type=int, default=64)
    parser.add_argument("--hidden_conv2", type=int, default=32)
    parser.add_argument("--noise_type", type=str, default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2042)

    training_config = dict()
    parser=parser.parse_args(model_detail)
    for arg in vars(parser):
        training_config[arg] = getattr(parser, arg)
    
    return training_config

class ROLANDGNN(torch.nn.Module):
    def __init__(self, input_dim, device, mlp_hidd: tuple[int], conv_hidd: tuple[int], dropout=0.0, update='moving'):
        
        super(ROLANDGNN, self).__init__()
        #Architecture: 
            #2 MLP layers to preprocess BERT repr, 
            #2 GCN layer to aggregate node embeddings
            #HadamardMLP as link prediction decoder
        
        #You can change the layer dimensions but 
        #if you change the architecture you need to change the forward method too
        #TODO: make the architecture parameterizable

        linear_hidd1, linear_hidd2 = mlp_hidd

        self.preprocess1 = Linear(input_dim, linear_hidd1).to(device)
        self.preprocess2 = Linear(linear_hidd1, linear_hidd2).to(device)
        self.full_connect = Linear(input_dim, linear_hidd2).to(device)

        conv_hidden1, conv_hidden2 = conv_hidd

        self.conv1 = GCNConv(linear_hidd2, conv_hidden1).to(device)
        self.conv2 = GCNConv(conv_hidden1, conv_hidden2).to(device)
        

        self.encoder1, self.encoder2 = None, None
        self.encoder_ware = None
        # Initialize the loss function to BCEWithLogitsLoss
        # self.loss_fn = loss(reduction="mean")
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.dropout = dropout
        self.update = update
        if update=='moving':
            self.tau = torch.Tensor([0]).to(device)
        elif update=='learnable':
            self.tau = torch.nn.Parameter(torch.Tensor([0])).to(device)
        elif update=='gru':
            self.gru1 = GRUCell(conv_hidden1, conv_hidden1).to(device)
            self.gru2 = GRUCell(conv_hidden2, conv_hidden2).to(device)
        elif update=='mlp':
            self.mlp1 = Linear(conv_hidden1*2, conv_hidden1).to(device)
            self.mlp2 = Linear(conv_hidden2*2, conv_hidden2).to(device)
        else:
            assert(update>=0 and update <=1)
            self.tau = torch.Tensor([update])
        self.previous_embeddings = None # [torch.zeros((num_nodes, conv_hidden1)).cuda(), torch.zeros((num_nodes, conv_hidden2)).cuda()]
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
        features.weight = nn.Parameter(node_fea, requires_grad=False).cuda()

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

    def forward(self, x, edge_index, graphsage: bool = False, gcn_only: bool=False, previous_embeddings=None):
        
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
            x = neighbor_emb.t()
        # return scores, current_embeddings

        # do some combination
        if not gcn_only:
            h = self.preprocess1(x)
            h = F.leaky_relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, inplace=True)
            h = self.preprocess2(h)
            h = F.leaky_relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, inplace=True)
        else:
            h = self.full_connect(x)

        #GRAPHCONV
        #GraphConv1
        h = self.conv1(h, edge_index)
        h = F.leaky_relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout,inplace=True) 
        prev_emb_norm = torch.norm(self.previous_embeddings[0], p=2, dim=1, keepdim=True)
        if not torch.sum(prev_emb_norm).item():
            prev_emb_norm = torch.ones(prev_emb_norm.shape).cuda()

        # first align the embedding in the same row length
        row_tranucate = self.previous_embeddings[0].shape[0]
        past_emb = torch.zeros((h.shape[0], h.shape[1])).cuda()
        try:
            past_emb[:row_tranucate] = self.previous_embeddings[0]
        except Exception as e:
            print("the error occur at past embedding", e)
            print("the shape of past embedding", past_emb.shape)
            print("the shape of previous embedding", self.previous_embeddings[0].shape)
            sys.exit(0)

        if self.update=='gru':
            h = self.gru1(h, past_emb) 
        elif self.update=='mlp':
            h = torch.cat((h, past_emb), dim=1) # /prev_emb_norm
            h = self.mlp1.forward(h)
        else:
            h = torch.Tensor(self.tau * self.previous_embeddings[0].clone() + (1-self.tau) * h.clone())
                
        h_norm = torch.norm(h, p=2, dim=1, keepdim=True)
        h = h/h_norm

        current_embeddings[0] = h.clone().detach()
        #GraphConv2
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, inplace=True) # 
        
        # normalization of past embedding
        conv2_prev_norm = torch.norm(self.previous_embeddings[1], p=2, dim=1, keepdim=True)
        if not torch.sum(conv2_prev_norm).item():
            conv2_prev_norm = torch.ones(conv2_prev_norm.shape).cuda()

        past_emb2 = torch.zeros((h.shape[0], h.shape[1])).cuda()
        past_emb2[:row_tranucate] = self.previous_embeddings[1]

        #Embedding Update after second layer
        if self.update=='gru':
            h = self.gru2(h, past_emb2)  
        elif self.update=='mlp':
            h = torch.cat((h, past_emb2), dim=1) 
            h = self.mlp2(h)
        else:
            h = torch.Tensor(self.tau * self.previous_embeddings[1].clone() + (1-self.tau) * h.clone())

        h_norm = torch.norm(h, p=2, dim=1, keepdim=True)
        h = h/h_norm

        current_embeddings[1] = h.detach()

        return h, current_embeddings
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    