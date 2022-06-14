"""
Author: Guang Yang
Dynamic GCN class 
"""
import sys
import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

sys.path.insert(0, '/home/ec2-user/SageMaker/NHLShotQuality/src/')
#sys.path.insert(0, '/home/ec2-user/SageMaker/NHLShotQuality/src/features/')
from torch_scatter import scatter_mean
from torch_scatter import scatter_max
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import SAGEConv
# from torch_geometric.nn import GINConv


from utils import save_json_file  # Attetnion Weights


# References
# DynamicRumorGCN - https://github.com/JihoChoi/dynamic-gcn
# RvNN - https://github.com/majingCUHK/Rumor_RvNN
# BiGCN - https://github.com/TianBian95/BiGCN/
# Self-Attention - https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py


class TDGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        
    def forward(self, data):
        """add edge weight but without skip connection"""
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        #x1 = copy.copy(x.float())
        #x = self.conv1(x, edge_index)
        #print(x.shape)
        #print(edge_index.shape)
        #print(edge_weight.shape)
        x = self.conv1(x, edge_index, edge_weight=edge_weight) #error
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        # READOUT LAYER: mean, max pooling (nodes -> graph)
        """
        x_mean = scatter_mean(x, data.batch, dim=0)  # B x 64
        x_max = scatter_max(x, data.batch, dim=0)[0]  # B x 64
        x = torch.cat((x_mean, x_max), 1)  # CONCAT(mean, max)
        return x  # B x 128
        """
        x_mean = scatter_mean(x, data.batch, dim=0)  # B x num_of_edges
        return x_mean
        
class BUGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        
    def forward(self, data):
        """add edge weight but without skip connection"""
        x, edge_index, edge_weight = data.x, data.BU_edge_index, data.BU_edge_weight
        #x1 = copy.copy(x.float())
        #x = self.conv1(x, edge_index)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        # READOUT LAYER: mean, max pooling (nodes -> graph)
        """
        x_mean = scatter_mean(x, data.batch, dim=0)  # B x 64
        x_max = scatter_max(x, data.batch, dim=0)[0]  # B x 64
        x = torch.cat((x_mean, x_max), 1)  # CONCAT(mean, max)
        return x  # B x 128
        """
        x_mean = scatter_mean(x, data.batch, dim=0)  # B x 64
        return x_mean
    
class BiGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BiGCN, self).__init__()
        self.TDGCN = TDGCN(in_feats, hid_feats, out_feats)
        self.BUGCN = BUGCN(in_feats, hid_feats, out_feats)
        #self.TDRumorGCN = TDRumorGCN(in_feats, hid_feats, out_feats)
        #self.BURumorGCN = BURumorGCN(in_feats, hid_feats, out_feats)

    def forward(self, data):
        TD_x = self.TDGCN(data) #error
        BU_x = self.BUGCN(data)
        x = torch.cat((TD_x, BU_x), 1)
        return x

class Network(nn.Module):
    # def __init__(self, in_feats, hid_feats, out_feats, snapshot_num, device):
    def __init__(self, in_feats, hid_feats, out_feats, settings):
        super(Network, self).__init__()

        Network.snapshot_num = settings['snapshot_num']
        Network.device = settings['cuda']
        Network.learning_sequence = settings['learning_sequence']

        self.GCN_0 = BiGCN(in_feats, hid_feats, out_feats)
        self.W_s1 = nn.Linear(out_feats * 2 * 2, 1)  # additive attention
        self.fc = nn.Linear(out_feats * 2, 2)
        #self.fc = nn.Linear((out_feats + hid_feats) * 2, 2)
        # self.fc = nn.Linear((out_feats + hid_feats) * 2 * 2, 4)
        # self.fc = nn.Linear((out_feats + hid_feats) * 2 * 4, 4)
        self.init_weights()

    def init_weights(self):  # Xavier Init
        init.xavier_normal_(self.GCN_0.TDGCN.conv1.weight)
        init.xavier_normal_(self.GCN_0.TDGCN.conv2.weight)
        init.xavier_normal_(self.GCN_0.BUGCN.conv1.weight)
        init.xavier_normal_(self.GCN_0.BUGCN.conv2.weight)
        init.xavier_normal_(self.W_s1.weight)
        init.xavier_normal_(self.fc.weight)

    def append_results(self, string):  # TODO:
        with open("/home/ec2-user/SageMaker/NHLShotQuality/models/attention.txt", 'a') as out_file:
            out_file.write(str(string) + '\n')

    def additive_attention(self, x):  # TODO:
        x_stack = torch.stack(x, 1)  # B * S * 256
        x_context = x_stack.mean(dim=1)  # B * S

        attn_w = []
        for current_x in x:  # TODO: BATCH PARALLEL
            attn_w.append(self.W_s1(torch.cat((current_x, x_context), 1)))
        attn_weights = torch.cat((attn_w), 1)  # B * S
        attn_weights = F.softmax(attn_weights, dim=1)  # B * S
        updated_x = []

        # print(attn_weights)  # attention # TODO:
        for index, current_x in enumerate(x):
            weighted_x = torch.bmm(
                current_x.unsqueeze(2),  # B * 256 * 1
                attn_weights[:, index].unsqueeze(1).unsqueeze(2)  # B * 1 * 1
            )
            updated_x.append(weighted_x.squeeze(2))
        updated_x = torch.stack(updated_x, 1)
        return updated_x


    def dot_product_attention(self, query, key, value, mask=None):  # self-attention
        dk = query.size()[-1]  # 256
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)

        # self.append_results(attention.data)  # batch average  # TODO:
        return attention.matmul(value)


    def attention_module(self, x):  # TODO: REFACTORING
        # x: Batch x Seq x Embedding - E.g.: (20, 5, 256)

        if Network.learning_sequence == "mean":
            x_stack = torch.stack(x, 1)  # B x S x D - E.g.: (20, 5, 256)
            x = x_stack.mean(dim=1)

        elif Network.learning_sequence == "mean_max":
            x_stack = torch.stack(x, 1)  # B x S x D - E.g.: (20, 5, 256)
            x_mean = x_stack.mean(dim=1)
            x_max = torch.max(x_stack, dim=1)[0]
            x = torch.cat((x_mean, x_max), 1)  # CONCAT(mean, max)

        elif Network.learning_sequence == "additive":
            x_stack = self.additive_attention(x)
            x = x_stack.mean(dim=1)

        elif Network.learning_sequence == "dot_product":
            x_stack = torch.stack(x, 1)  # B x S x D - E.g.: (20, 5, 256)
            x_stack = self.dot_product_attention(x_stack, x_stack, x_stack)
            x = x_stack.mean(dim=1)

        elif Network.learning_sequence == "LSTM":
            # TODO: include in refactoring scripts
            pass

        elif Network.learning_sequence == "GRU":
            # TODO: include in refactoring scripts
            pass

        else:
            pass

        return x

    def forward(self, snapshots):

        # 2) GCN LAYERS + 3) READOUT LAYER
        x = []
        for s in snapshots:
            #print(self.GCN_0(s).shape)  #error
            x.append(self.GCN_0(s))

        # 4) ATTENTION LAYER
        x = self.attention_module(x)
        #print(x.shape)

        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x