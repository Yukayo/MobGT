import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import torch.optim as optim

import os
import json
import time
import argparse
import numpy as np
import copy
import pandas as pd
import math
import pickle
from json import encoder
from torch_geometric.nn import GCNConv
from torch.nn import Parameter

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

from torch.utils.data import DataLoader
import os.path as osp
from collections import deque, Counter
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

import algos

def generate_queue(train_idx, mode, mode2):
    user = list(train_idx.keys())  # 获取所有用户ID
    train_queue = deque()
    np.random.seed(1)
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])  # [1:]
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue

class Foursquare(InMemoryDataset):
    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        assert split in ['train', 'test']
        super(Foursquare, self).__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'test.pickle', 'train.index', 'test.index'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'whole'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        if os.getcwd() == '/fast/xuxh' or os.getcwd() == '/fast':
            os.chdir('/fast/xuxh/graphormer_new')
        for split in ['train', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            with open(osp.join(self.raw_dir, f'{split}_idx.pkl'), 'rb') as f:
                dict1 = pickle.load(f)
                if split == 'train_OD' or split=='toyota_net_vis':
                    split = 'train'
                if split == 'test_OD':
                    split = 'test'
                indices = generate_queue(dict1, 'random', split)

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')
            if split == "train":
                self.train_indices = indices
            else:
                self.test_indices = indices

            data_list = []
            pp = 0
            for idx in indices:
                # pp += 1
                # if idx[1]==len(mols[idx[0]]) and split=="train":
                #     continue
                # if split=="train":
                #     mol = mols[idx[0]][idx[1]+1]
                # else:
                #     mol = mols[idx[0]][idx[1]]
                mol = mols[idx[0]][idx[1]]
                x = mol['node_name'].to(torch.long).view(-1, 1)
                y = mol['target'].to(torch.long)

                adj = mol['edge_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_item(item):

    num_virtual_tokens = 1
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    if edge_attr is None:
        edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)
        # edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.float)

    N = x.size(0)

    x = convert_to_single_emb(x)  # For ZINC: [n_nodes, 1]

    # node adj matrix [N, N] bool
    adj_orig = torch.zeros([N, N], dtype=torch.bool)
    adj_orig[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here

    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.float)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )  # [n_nodes, n_nodes, 1] for ZINC

    shortest_path_result, path = algos.floyd_warshall(
        adj_orig.numpy()
    )  # [n_nodesxn_nodes, n_nodesxn_nodes]
    max_dist = np.amax(shortest_path_result)
    # path=path.astype(np.float32)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    # rel_pos = torch.from_numpy((shortest_path_result)).float()
    attn_bias = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.float
    )  # with graph token

    adj = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.bool
    )
    adj[edge_index[0, :], edge_index[1, :]] = True

    for i in range(num_virtual_tokens):
        adj[N + i, :] = True
        adj[:, N + i] = True

    # for i in range(N + num_virtual_tokens):
    #     for j in range(N + num_virtual_tokens):

    #         val = True if random.random() < 0.3 else False
    #         adj[i, j] = adj[i, j] or val

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj_orig.long().sum(dim=1).view(-1)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.adj = adj

    return item

class MyFoursquareDataset(Foursquare):
    def download(self):
        super(MyFoursquareDataset, self).download()

    def process(self):
        super(MyFoursquareDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int): #判断是不是int
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)

dataset = {
        # "num_class": 1,
        # loss和deepmove暂且一致
        "loss_fn": nn.NLLLoss().cuda(),
        # "metric": "cross_entropy",
        "metric": "NLLLoss",
        "metric_mode": "min",
        "train_dataset": MyFoursquareDataset(
            subset=False, root="../dataset/toyota", split="train"
        ),
        "test_dataset": MyFoursquareDataset(
            subset=False, root="../dataset/toyota", split="test"
        ),
        # "max_node": 2708,
    }


# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.lin = torch.nn.Linear(in_channels, out_channels)
 
#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
 
#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
 
#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x)
 
#         # Step 3: Compute normalization.
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
 
#         # Step 4-5: Start propagating messages.
#         return self.propagate(edge_index, x=x, norm=norm)
 
#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]
 
#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j

# class Net(torch.nn.Module):
#     # torch.nn.Module 是所有神经网络单元的基类
#     def __init__(self,use_bias=True):
#         super(Net, self).__init__()  ###复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
#         self.use_bias = use_bias
#         self.weight = nn.Parameter(torch.Tensor(dataset.num_node_features, dataset.num_classes))
#         if self.use_bias:
#             self.bias = nn.Parameter(torch.Tensor(dataset.num_classes))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
 
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)
 
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight)
#         if self.use_bias:
#             nn.init.zeros_(self.bias)
 
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
 
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
 
#         # if self.use_bias:
#         #     x += self.bias
 
#         return F.log_softmax(x, dim=1)



# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GCNConv(19800, 500)
#         self.conv2 = GCNConv(500, 19800)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)

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

def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN().to(device)

data = dataset['train_dataset'].data.to(device)

gcn_nhid=[32,64]
poi_embed_dim=128
gcn_dropout=0.3

gcn_nfeat = X.shape[1]
poi_embed_model = GCN(ninput=gcn_nfeat,
                        nhid=gcn_nhid,
                        noutput=poi_embed_dim,
                        dropout=gcn_dropout).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(20):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()

outputs = model(dataset['test_dataset'])
queue_len = len(outputs)
users_acc = {}
for u in range(queue_len):
    y_pred = outputs[u]["y_pred"]
    y_true = outputs[u]["y_true"]
    if u not in users_acc:
        users_acc[u] = [0, 0, 0]
        users_acc[u][0] += len(y_true)
        acc = get_acc(y_true, y_pred)
        users_acc[u][1] += acc[2]
        users_acc[u][2] += acc[1]
    tmp_acc = [0.0,0.0]
    sum_test_samples = 0.0
    for u in users_acc:
        tmp_acc[0] = users_acc[u][1] + tmp_acc[0]
        tmp_acc[1] = users_acc[u][2] + tmp_acc[1]
        sum_test_samples = sum_test_samples + users_acc[u][0]
    avg_acc = (np.array(tmp_acc)/sum_test_samples)
    fin_acc=float(avg_acc[0])
    fin_acc1=float(avg_acc[1])






print(f'Accuracy@0: {fin_acc:.4f}')
print(f'Accuracy@5: {fin_acc1:.4f}')

