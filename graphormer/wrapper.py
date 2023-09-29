# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import random
import torch
import numpy as np
import torch_geometric.datasets
import copy
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
from owndata import Foursquare,FoursquareGraph,ToyotaGraph
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos


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
    # N=item.edge_attr.size(0)

    x = convert_to_single_emb(x)  # For ZINC: [n_nodes, 1]

    user = convert_to_single_emb(item.user)

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


    adj1 = torch.zeros(
        [N , N], dtype=torch.bool
    )
    adj1[edge_index[0, :], edge_index[1, :]] = True


    for i in range(num_virtual_tokens):
        adj[N + i, :] = True
        adj[:, N + i] = True

    # for i in range(N + num_virtual_tokens):
    #     for j in range(N + num_virtual_tokens):

    #         val = True if random.random() < 0.3 else False
    #         adj[i, j] = adj[i, j] or val

    # combine
    item.user=user
    item.adj1=adj1
    item.x = x
    # item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj_orig.long().sum(dim=1).view(-1)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.adj = adj

    return item


class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(MyGraphPropPredDataset, self).download()

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyPygPCQM4MDataset(PygPCQM4MDataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)

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

class MyFoursquareGraphDataset(FoursquareGraph):
    def download(self):
        super(MyFoursquareGraphDataset, self).download()

    def process(self):
        super(MyFoursquareGraphDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int): #判断是不是int
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)

class MyGowallaGraphDataset(FoursquareGraph):
    def download(self):
        super(MyGowallaGraphDataset, self).download()

    def process(self):
        super(MyGowallaGraphDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int): #判断是不是int
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


# ADDED MyToyotaGraphDataset from graphormer_new_cuda2
class MyToyotaGraphDataset(ToyotaGraph):
    def download(self):
        super(MyToyotaGraphDataset, self).download()

    def process(self):
        super(MyToyotaGraphDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int): #判断是不是int
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)
