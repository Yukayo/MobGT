import torch
from torch.nn import Linear, Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

import torch.nn.functional as F
import pandas as pd
import math
import copy
from torch_geometric.nn import GCNConv, GATConv
from data import GraphDataModule, get_dataset
from argparse import ArgumentParser
from data import GraphDataModule_train
from torch_geometric.utils import to_undirected
from torch_geometric.utils import add_self_loops, degree
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse.linalg import eigsh

from collator import collator,collator1
from wrapper import (
    MyGraphPropPredDataset,
    MyPygPCQM4MDataset,
    MyFoursquareDataset
)

import numpy as np

from pytorch_lightning import LightningDataModule

import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial




# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.



def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0,我感觉不需要pad
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_1d_unsqueeze1(x, padlen):
    # x = x + 1  # pad id = 0,我感觉不需要pad
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze1(x, padlen):
    # x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze2(x, padlen):
    # x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    # print(xdim,padlen)
    if xdim < padlen:
        new_x = x.new_zeros([2, padlen], dtype=x.dtype)
        new_x[:,:xdim] = x
        # x = new_x
        return new_x.unsqueeze(0)
    return x

def pad_time_unsqueeze(x, padlen): 
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_bool(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(False)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze1(x, padlen1, padlen2, padlen3):
    # x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_index_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)
       

class Batch2:
    def __init__(
        self,
        idx,
        attn_bias,
        attn_edge_type,
        rel_pos,
        in_degree,
        out_degree,
        x,
        edge_input,
        y,
        adj,
        adj1,
        time,
        feature_matrix,
        edge_index
    ):
        super(Batch2, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.rel_pos = (
            attn_bias,
            attn_edge_type,
            rel_pos,
        )
        self.edge_input = edge_input
        self.adj = adj
        self.time=time
        self.adj1=adj1
        self.feature_matrix=feature_matrix
        self.edge_index=edge_index

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = (
            self.in_degree.to(device),
            self.out_degree.to(device),
        )
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.attn_edge_type, self.rel_pos = (
            self.attn_bias.to(device),
            self.attn_edge_type.to(device),
            self.rel_pos.to(device),
        )
        self.edge_input = self.edge_input.to(device)
        self.adj = self.adj.to(device)
        self.time=self.time.to(device)
        self.adj1=self.adj1.to(device)
        self.feature_matrix=self.feature_matrix.to(device)
        self.edge_index=self.edge_index.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)

def collator2(items, max_node=512, multi_hop_max_dist=20, rel_pos_max=20):

    num_virtual_tokens = 1
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.rel_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.adj,
            item.time,
            item.adj1,
            item.edge_index
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        rel_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        adjs,
        time,
        adjs1,
        edge_indexs
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):

        attn_biases[idx][num_virtual_tokens:, num_virtual_tokens:][
            rel_poses[idx] >= rel_pos_max
        ] = float("-inf")

    # Maximum number of nodes in the batch.
    max_node_num = max(i.size(0) for i in xs)
    #我看不懂下面两句什么目的，暂时注释掉
    div = max_node_num // 4
    max_node_num = 4 * div + 3

    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze1(i, max_node_num) for i in xs]) #原句
    
    # times=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time]) #for max
    times=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time]) #for avg
    # x = torch.cat([i for i in xs])

    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [
            pad_attn_bias_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_biases
        ]
    )
    adj = torch.cat([pad_2d_bool(i, max_node_num + num_virtual_tokens) for i in adjs])


    adj1 = torch.cat([pad_2d_bool(i, max_node_num) for i in adjs1])


    #特征值
    edge_attr = adj1.long()
    edge_degree = torch.diag_embed(edge_attr.sum(dim=-1))
    edge_laplace=edge_degree-edge_attr

    # 求解特征值与特征向量
    for i in range(len(edge_laplace)):
        if i==0:
            # t=torch.eig(edge_laplace[i].float(),eigenvectors=True)[1]
            t=torch.linalg.eig(edge_laplace[i].float())[1]
        else:
            # t=torch.cat([t, torch.eig(edge_laplace[i].float(),eigenvectors=True)[1]],dim=0)
            t=torch.cat([t, torch.linalg.eig(edge_laplace[i].float())[1]],dim=0)
    feature_matrix=t.reshape(len(adj),t.size(1),t.size(1))
    flag=0
    temp=[]
    
    for i in edge_indexs:
        temp.append(pad_2d_unsqueeze2(i, max_node_num))
        
        # edge_index=torch.cat([
        #     pad_2d_unsqueeze2(i, max_node_num)
        #     for i in edge_indexs
        #     ])
    edge_index=torch.cat(temp)

    attn_edge_type = torch.cat(
        [
            pad_edge_type_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_edge_types
        ]
    )
    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])
    return Batch2(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        rel_pos=rel_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        edge_input=edge_input,
        y=y,
        adj=adj,
        time=times,
        adj1=adj1,
        feature_matrix=feature_matrix,
        edge_index=edge_index
    )

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, torch.Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0   
    else:
        return max(edge_index.size(0), edge_index.size(1))

def add_remaining_self_loops(edge_index,
                             edge_weight = None,
                             fill_value: float = 1.,
                             num_nodes= None):
    r"""
    如果原图已经存在部分自环或已加权则保持不变
    仅会为没有自环并且未加权的自环边根据 fill_value 填充权重值
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): 一维向量, 表示边的权重
        fill_value (float, optional): 权重默认填充值 (默认值为 1)
        num_nodes (int, optional): 图中的节点数量

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    # edge_index    (2,E) 存储边
    # edge_weight   (E) 边的个数 [1,1,...,1]
    # fill_value    填充值, 默认为 1
    # num_nodes     节点个数 -> maybe_num_nodes
    N = maybe_num_nodes(edge_index, num_nodes)  # 节点个数
    row, col = edge_index[0], edge_index[1]     # 边起始点(row)和终点(col)索引列表
    mask = row != col   # 非对角线连接为 True -> 1  # True/False 列表判断是否为自环

    # loop_index:torch.Size([2708])
    # tensor([   0,    1,    2,  ..., 2705, 2706, 2707], device='cuda:0')
    loop_index = torch.arange(0, N, dtype=row.dtype, device=row.device) # 定义添加自环的所需的索引
    # loop_index:torch.Size([2, 2708])
    # tensor([[   0,    1,    2,  ..., 2705, 2706, 2707],
    # [   0,    1,    2,  ..., 2705, 2706, 2707]], device='cuda:0')
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)   # 定义添加自环的所需的索引
    # edge_index:torch.Size([2, 13264])
    # 根据边关系的存储形式, 拼接即相当于添加自环
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)    # 非对角线元素 + 自环元素

    if edge_weight is not None:
        inv_mask = ~mask    # 按位取反 -> 对角线连接为 True(1)  # inv_mask:10556
        # loop_weight:torch.Size([2708])
        loop_weight = torch.full((N, ), fill_value, dtype=edge_weight.dtype,    # 定义自环的权重
                                 device=edge_index.device)  
        
        # edge_weight:torch.Size([10556])
        # remaining_edge_weight:torch.Size([0])
        remaining_edge_weight = edge_weight[inv_mask]   # 原始边关系中自环的权重
        if remaining_edge_weight.numel() > 0:           # 如果原始边有自环则继承原始自环权重
            loop_weight[row[inv_mask]] = remaining_edge_weight
        # 根据边关系的存储形式, 拼接即相当于说明自环的权重
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    # edge_index:torch.Size([2, 13264]) 非对角线元素 + 自环元素
    # edge_weight:torch.Size([13264])   非对角线元素 + 自环元素权重
    return edge_index, edge_weight


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 10495)

    def forward(self, data):
        x, edge_index = data.x.float().to(device), data.edge_index.squeeze().to(device)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# class GAT_NET(torch.nn.Module):
#     def __init__(self, features, hidden, classes, dropout, heads):
#         super(GAT_NET, self).__init__()
#         self.gat1 = GATConv(features, hidden[0], heads, dropout)  # 定义GAT层，使用多头注意力机制
#         self.gat2 = GATConv(hidden[1], classes, heads, dropout)  # 因为多头注意力是将向量拼接，所以维度乘以头数。

#     def forward(self, x, edge_index):
#         # x, edge_index = data.x, data.edge_index
#         x = self.gat1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.gat2(x, edge_index)
#         # x = F.relu(x)
#         # x = F.dropout(x, training=self.training)
#         # x = F.log_softmax(x, dim=1)
#         return x




class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    图注意力层
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征维度
        self.out_features = out_features   # 节点表示向量的输出特征维度
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)   # [N, out_features]
        N = h.size()[0]    # N 图的节点数
        
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        
        zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷
        attention = torch.where(adj>0, e, zero_vec)   # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout 
        
        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)   # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid[1], n_class, dropout=dropout,alpha=alpha, concat=False)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)   # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)   # dropout，防止过拟合
        x = self.out_att(x, adj)
        # x = F.relu(self.out_att(x, adj))   # 输出并激活
        # x = F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定
        return x 



def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat) #将ndarray解释为矩阵,取值方法adj_mat[x,y]
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat) #np.linalg.matrix_power求矩阵的幂次方，此处为求逆
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')

class GAT_GPT(torch.nn.Module):
    def __init__(self, num_layers, heads, dim_in, dim_hidden, dim_out, dropout):
        super(GAT_GPT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        self.attentions = torch.nn.ModuleList()
        self.fc = torch.nn.Linear(num_layers*heads*dim_hidden, dim_out)

        for i in range(num_layers):
            self.convs.append(GATConv(dim_in, dim_hidden, heads=heads, dropout=dropout))
            self.batchnorm.append(torch.nn.BatchNorm1d(dim_hidden*heads))
            self.attentions.append(torch.nn.Linear(dim_hidden*heads, 1))
            dim_in = dim_hidden*heads

    def forward(self, x, edge_index):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.batchnorm[i](x)
            x = torch.nn.functional.elu(x)
            attention = self.attentions[i](x)
            attention = torch.nn.functional.softmax(attention, dim=1)
            x = x * attention
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

torch.manual_seed(1)
torch.cuda.manual_seed(1)

path="/fast/xuxh/dataset/foursquaregraph/raw/"
# parser = ArgumentParser()
# parser = GraphDataModule.add_argparse_args(parser)
# args = parser.parse_args()

# print(args)

# ------------
# data
# ------------
# dm = GraphDataModule.from_argparse_args(args)

# train_data=GraphDataModule_train(path)

raw_A=pd.read_csv(path+"Graph_dist.csv").to_numpy()
raw_X=pd.read_csv(path+"Graph_poi.csv").to_numpy()

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

num_pois = raw_X.shape[0]
one_hot_encoder = OneHotEncoder()
cat_list = list(raw_X[:, 4])
one_hot_encoder.fit(list(map(lambda x: [x], cat_list))) #他把类别做成了独热向量
one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
num_cats = one_hot_rlt.shape[-1]

# X=copy.deepcopy(raw_X)

X = np.zeros((num_pois, raw_X.shape[-1] - 1 -1 + num_cats), dtype=np.float32)
X[:, 0] = raw_X[:, 0]
X[:, 1:num_cats + 1] = one_hot_rlt
X[:, num_cats + 1:] = raw_X[:, 2:-1] 

gat_nfeat = X.shape[1]
# gat_nhid=[32, 128]
gat_nhid= 512
gat_poi_embed_dim=320
gat_dropout=0.3
gat_alpha=0.2

edge_index = torch.LongTensor(raw_A).nonzero(as_tuple=False).t().contiguous().to(device)
edge_attr = torch.LongTensor(raw_A)[edge_index[0], edge_index[1]].to(device)

# edge_attr=torch.FloatTensor(edge_attr).to(device)
# edge_attr=edge_attr.to(torch.float).to(device)
# edge_index=add_remaining_self_loops(edge_index)
# edge_index=edge_index[0].to(device)
# edge_index=edge_index.float().to(device)

X=torch.FloatTensor(X).to(device)

gcn_nfeat = X.shape[1]
gcn_nhid=[32, 128]
gcn_poi_embed_dim=320
gcn_dropout=0.3


# A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
# A = torch.from_numpy(A)
# A = A.to(device=device, dtype=torch.float)

raw_A = torch.LongTensor(raw_A).to(device)

# poi_embed_model = GCN(ninput=gcn_nfeat,
#                         nhid=gcn_nhid,
#                         noutput=gcn_poi_embed_dim,
#                         dropout=gcn_dropout).to(device)

# poi_embed_model = GAT(n_feat=gat_nfeat,
#                           n_hid=gat_nhid,
#                           n_class=gat_poi_embed_dim,
#                           dropout=gat_dropout,
#                           alpha=gat_alpha,
#                           n_heads=3).to(device)
                          
# poi_embed_model = GAT_NET(features=gat_nfeat,
#                           hidden=gat_nhid,
#                           classes=gat_poi_embed_dim,
#                           dropout=gat_dropout,
#                           heads=4).to(device)


poi_embed_model = GAT_GPT(num_layers=2,
                            heads=4,
                            dim_in=gat_nfeat,
                            dim_hidden=gat_nhid,
                            dim_out=gat_poi_embed_dim,
                            dropout=gat_dropout).to(device)


                          

optimizer = torch.optim.Adam(poi_embed_model.parameters(), lr=0.01, weight_decay=5e-4)

poi_embed_model.train()

for epoch in range(10):
    optimizer.zero_grad()
    # out=poi_embed_model(X, raw_A)
    out=poi_embed_model(X, edge_index)
    # out=poi_embed_model(X, A)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('GAT Accuracy: {:.4f}'.format(acc))