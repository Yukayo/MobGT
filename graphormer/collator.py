# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import numpy as np
import pickle

POI_DATA_DIR = os.path.join("..", "dataset", "poi_data")

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0,我感觉不需要pad
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

def pad_2d_squeeze(x, padlen):
    x = x - 1  # pad id = 0
    x = x
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

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

def pad_index_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

class Batch:
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
    ):
        super(Batch, self).__init__()
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
        return self

    def __len__(self):
        return self.in_degree.size(0)


class Batch1:
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
        time_normal,
        user,
        cat,
        poi_pos
    ):
        super(Batch1, self).__init__()
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
        self.time_normal=time_normal
        self.user=user
        self.cat=cat
        self.poi_pos=poi_pos

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
        self.time_normal=self.time_normal.to(device)
        self.user=self.user.to(device)
        self.cat=self.cat.to(device)
        self.poi_pos=self.poi_pos.to(device)


        return self

    def __len__(self):
        return self.in_degree.size(0)

#for pure graphormer
def collator(items, max_node=512, multi_hop_max_dist=20, rel_pos_max=20):

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
    # y = torch.cat(ys)
    y = torch.cat([i+1 for i in ys])
    x = torch.cat([pad_2d_squeeze(i, max_node_num) for i in xs]) #原句

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
    attn_edge_type = torch.cat(
        [
            pad_edge_type_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_edge_types
        ]
    )
    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])
    return Batch(
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
    )

def freedman_diaconis_bins(x, return_bins=False):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    binsize = 2 * iqr * np.power(len(x), -1/3)
    bins = np.ceil((np.max(x) - np.min(x)) / binsize)
    if return_bins:
        return int(bins), np.histogram(x, int(bins))[1]
    else:
        return int(bins)

def collator_foursquare(items, max_node=512, multi_hop_max_dist=20, rel_pos_max=20):

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
            item.edge_index,
            item.time_normal,
            item.user,
            item.cat
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
        edge_indexs,
        time_normals,
        users,
        cats
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):

        attn_biases[idx][num_virtual_tokens:, num_virtual_tokens:][
            rel_poses[idx] >= rel_pos_max
        ] = float("-inf")

    # Maximum number of nodes in the batch.
    max_node_num = max(i.size(0) for i in xs)
    #我看不懂下面两句什么目的，暂时注释掉
    # div = max_node_num // 4
    # max_node_num = 4 * div + 3

    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_squeeze(i, max_node_num) for i in xs]) #原句

    # times=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time]) #for max
    times=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time]) #for avg
    time_normal=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time_normals])
    user=torch.cat(users)
    cat=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in cats])

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

    attn_edge_type = torch.cat(
        [
            pad_edge_type_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_edge_types
        ]
    )

    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])

    mask = x == 0
    mask = ~mask
    indx = mask.sum(dim=-2)


    poi_pos= torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    poi_distance_matrix = pickle.load(open(os.path.join(POI_DATA_DIR, "tky_distance.pkl"), 'rb'), encoding='iso-8859-1')
    distance_matrix_row=np.delete(poi_distance_matrix,0,axis=0)
    # np.delete(distance_matrix,0,axis=1)
    distance_matrix = np.delete(distance_matrix_row,0,axis=1)
    distances_adjusted = distance_matrix - distance_matrix.min()
    num_bins,bins = freedman_diaconis_bins(distances_adjusted, True)
    for i in range(x.size(0)):
        for j in range(indx[i]):
            poi_pos[i][j][:indx[i]]=torch.LongTensor(np.digitize([poi_distance_matrix[x[i][j]][x[i][k]] for k in range(indx[i])],bins))


    return Batch1(
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
        time_normal=time_normal,
        user=user,
        cat=cat,
        poi_pos=poi_pos
    )
    
def collator_gowalla(items, max_node=512, multi_hop_max_dist=20, rel_pos_max=20):

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
            item.edge_index,
            item.time_normal,
            item.user,
            item.cat
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
        edge_indexs,
        time_normals,
        users,
        cats
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):

        attn_biases[idx][num_virtual_tokens:, num_virtual_tokens:][
            rel_poses[idx] >= rel_pos_max
        ] = float("-inf")

    # Maximum number of nodes in the batch.
    max_node_num = max(i.size(0) for i in xs)
    #我看不懂下面两句什么目的，暂时注释掉
    # div = max_node_num // 4
    # max_node_num = 4 * div + 3

    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_squeeze(i, max_node_num) for i in xs]) #原句

    # times=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time]) #for max
    times=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time]) #for avg
    time_normal=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time_normals])
    user=torch.cat(users)
    cat=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in cats])

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

    attn_edge_type = torch.cat(
        [
            pad_edge_type_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_edge_types
        ]
    )

    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])

    mask = x == 0
    mask = ~mask
    indx = mask.sum(dim=-2)


    poi_pos= torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    poi_distance_matrix = pickle.load(open(os.path.join(POI_DATA_DIR, "gowalla_distance.pkl"), 'rb'), encoding='iso-8859-1')
    distance_matrix_row=np.delete(poi_distance_matrix,0,axis=0)
    # np.delete(distance_matrix,0,axis=1)
    distance_matrix = np.delete(distance_matrix_row,0,axis=1)
    distances_adjusted = distance_matrix - distance_matrix.min()
    num_bins,bins = freedman_diaconis_bins(distances_adjusted, True)
    for i in range(x.size(0)):
        for j in range(indx[i]):
            poi_pos[i][j][:indx[i]]=torch.LongTensor(np.digitize([poi_distance_matrix[x[i][j]][x[i][k]] for k in range(indx[i])],bins))


    return Batch1(
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
        time_normal=time_normal,
        user=user,
        cat=cat,
        poi_pos=poi_pos
    )

def collator_toyota(items, max_node=512, multi_hop_max_dist=20, rel_pos_max=20):

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
            item.edge_index,
            item.time_normal,
            item.user,
            item.cat
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
        edge_indexs,
        time_normals,
        users,
        cats
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):

        attn_biases[idx][num_virtual_tokens:, num_virtual_tokens:][
            rel_poses[idx] >= rel_pos_max
        ] = float("-inf")

    # Maximum number of nodes in the batch.
    max_node_num = max(i.size(0) for i in xs)
    #我看不懂下面两句什么目的，暂时注释掉
    # div = max_node_num // 4
    # max_node_num = 4 * div + 3

    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_squeeze(i, max_node_num) for i in xs]) #原句

    # times=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time]) #for max
    times=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time]) #for avg
    time_normal=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in time_normals])
    user=torch.cat(users)
    cat=torch.cat([pad_time_unsqueeze(i, max_node_num) for i in cats])

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
            t=torch.linalg.eig(edge_laplace[i].float())[1]
        else:
            t=torch.cat([t, torch.linalg.eig(edge_laplace[i].float())[1]],dim=0)
    feature_matrix=t.reshape(len(adj),t.size(1),t.size(1))

    attn_edge_type = torch.cat(
        [
            pad_edge_type_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_edge_types
        ]
    )

    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])

    mask = x == 0
    mask = ~mask
    indx = mask.sum(dim=-2)

    poi_pos= torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    poi_distance_matrix = pickle.load(open(os.path.join(POI_DATA_DIR, "toyota_distance.pkl"), 'rb'), encoding='iso-8859-1')
    distance_matrix_row=np.delete(poi_distance_matrix,0,axis=0)
    distance_matrix = np.delete(distance_matrix_row,0,axis=1)
    distances_adjusted = distance_matrix - distance_matrix.min()
    num_bins,bins = freedman_diaconis_bins(distances_adjusted, True)
    for i in range(x.size(0)):
        poi_pos[i][:indx[i]][:,:indx[i]]=torch.LongTensor([np.digitize([poi_distance_matrix[x[i][j]][x[i][k]] for k in range(indx[i])],bins) for j in range(indx[i])])

    return Batch1(
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
        time_normal=time_normal,
        user=user,
        cat=cat,
        poi_pos=poi_pos
    )
