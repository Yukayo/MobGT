# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data import get_dataset
from lr import PolynomialDecayLR
import torch
import math
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
import pickle
from torch.autograd import Variable

from utils.flag import flag_bounded

from scipy import sparse as sp

from modelGNN import GCN
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import OneHotEncoder
# import dgl

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction


    def forward(self, input, target):
        ce_loss = F.nll_loss(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.weight is not None:
            alpha = self.weight[target]
            focal_loss = alpha * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_acc(target, scores):
    """target and scores are torch cuda Variable
    return:
        acc[0]: Accuracy for top-10 predictions
        acc[1]: Accuracy for top-5 predictions
        acc[2]: Accuracy for top-1 predictions
        acc[3]: Accuracy for top-20 predictions
        ndcg[0]: NDCG for top-10 predictions
        ndcg[1]: NDCG for top-5 predictions
        ndcg[2]: NDCG for top-1 predictions
        ndcg[3]: NDCG for top-20 predictions
    """
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(20, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((4, 1))
    ndcg = np.zeros((4, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t != 0:
            if t in p[:20] and t > 0:
                acc[3] += 1
                rank_list = list(p[:20])
                rank_index = rank_list.index(t)
                ndcg[3] += 1.0 / np.log2(rank_index + 2)
            if t in p[:10] and t > 0:
                acc[0] += 1
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                acc[1] += 1
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                acc[2] += 1
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)            
        else:
            break
    return acc, ndcg

def get_acc1(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10)
    predx = idxx.cpu().numpy()
    acc= np.zeros((3, 1))
    ndcg= np.zeros((3, 1))
    for j in range(len(predx)):
        p=predx[j][0]
        t = target[j]
        if t != 0:
            if t in p[:10] and t > 0:
                acc[0] += 1
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                acc[1] += 1
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                acc[2] += 1
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        else:
            break
    return acc, ndcg

def MRR_metric(target, scores):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    y_true = target.data.cpu().numpy()
    y_pred = scores.data.cpu().numpy()
    mrr=0
    for j in range(len(y_pred)):
        rec_list = y_pred[j].argsort()[-len(y_pred[j]):][::-1]
        r_idx = np.where(rec_list == y_true[j])[0][0]
        mrr+=1 / (r_idx + 1)
    return mrr

def laplace_pe(adj):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    edge_attr = adj.long()
    edge_degree = torch.diag_embed(edge_attr.sum(dim=-1))
    edge_laplace=edge_degree-edge_attr

    # 求解特征值与特征向量
    for x in range(len(edge_laplace)):
        if x==0:
            # t=torch.eig(edge_laplace[x].float(),eigenvectors=True)[1]
            t=torch.linalg.eig(edge_laplace[x].float())[1]
        else:
            # t=torch.cat([t, torch.eig(edge_laplace[x].float(),eigenvectors=True)[1]],dim=0)
            t=torch.cat([t, torch.linalg.eig(edge_laplace[x].float())[1]],dim=0)
    feature_matrix=t.reshape(len(adj),t.size(1),t.size(1))           
    return feature_matrix


def init_bert_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        seq_len = history.size()[0]
        state_len = out_state.size()[0]
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = self.score(out_state[i], history[j]) #就是out_state /dot history q^t*W产生这样一个矩阵，最后再softmax归一化
        return F.softmax(attn_energies)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


class sin_PositionalEncoding(nn.Module):
    """
    正弦位置编码，即通过三角函数构建位置编码

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    """

    def __init__(self, dim: int, dropout: float, max_len=64):
        """
        :param dim: 位置向量的向量维度，一般与词向量维度相同，即d_model
        :param dropout: Dropout层的比率
        :param max_len: 句子的最大长度
        """
        # 判断能够构建位置向量
        if dim % 2 != 0:
            raise ValueError(f"不能使用 sin/cos 位置编码，得到了奇数的维度{dim:d}，应该使用偶数维度")

        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)  # 初始化pe
        position = torch.arange(1, max_len+1).unsqueeze(1) # 构建pos，为句子的长度，相当于pos
        for i in range(int(max_len/2)):
            if i ==0:
                pe[0::2][i] = torch.cos(position[i].float() * torch.exp((torch.arange(1, dim+1, 1, dtype=torch.float) * torch.tensor(
            (math.log(10000.0) / (torch.ones(dim)*(2*i+1))    )))))
                continue
            pe[1::2][i] = torch.sin(torch.exp((torch.arange(0, dim, 1, dtype=torch.float) * torch.tensor(
            (math.log(10000.0) / (torch.ones(dim)*(2*i))  )))))
            pe[0::2][i] = torch.cos(position[i].float() * torch.exp((torch.arange(1, dim+1, 1, dtype=torch.float) * torch.tensor(
            (math.log(10000.0) / (torch.ones(dim)*(2*i+1))    )))))
        pe = pe.unsqueeze(1)  # 扁平化成一维向量

        super(sin_PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)  # pe不是模型的一个参数，通过register_buffer把pe写入内存缓冲区，当做一个内存中的常量
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None,virtual='max'):
        """
        词向量和位置编码拼接并输出
        :param emb: 词向量序列（FloatTensor），``(seq_len, batch_size, self.dim)``
        :param step: 如果 stepwise("seq_len=1")，则用此位置的编码
        :return: 词向量和位置编码的拼接
        """
        # emb = emb * math.sqrt(self.dim)
        if step is None:
            # emb = emb + self.pe[:emb.size(0)]  # 拼接词向量和位置编码
            emb = self.pe[:emb.size(1)].squeeze().unsqueeze(0) / math.sqrt(self.dim)
            # emb = emb + self.proj(self.pe[:emb.size(1)].squeeze().unsqueeze(0))
        else:
            emb = emb + self.pe[step.squeeze()]/ math.sqrt(self.dim)
        # emb = self.drop_out(emb)
        return emb


class PositionalEncoding(nn.Module):
    """
    正弦位置编码，即通过三角函数构建位置编码

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    """

    def __init__(self, dim: int, dropout: float, max_len=5000):
        """
        :param dim: 位置向量的向量维度，一般与词向量维度相同，即d_model
        :param dropout: Dropout层的比率
        :param max_len: 句子的最大长度
        """
        # 判断能够构建位置向量
        if dim % 2 != 0:
            raise ValueError(f"不能使用 sin/cos 位置编码，得到了奇数的维度{dim:d}，应该使用偶数维度")

        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)  # 初始化pe
        position = torch.arange(0, max_len).unsqueeze(1)  # 构建pos，为句子的长度，相当于pos
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * torch.tensor(
            -(math.log(10000.0) / dim))))  # 复现位置编码sin/cos中的公式
        pe[:, 0::2] = torch.sin(position.float() * div_term)  # 偶数使用sin函数
        pe[:, 1::2] = torch.cos(position.float() * div_term)  # 奇数使用cos函数
        pe = pe.unsqueeze(1)  # 扁平化成一维向量

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)  # pe不是模型的一个参数，通过register_buffer把pe写入内存缓冲区，当做一个内存中的常量
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None, virtual='max'):
        """
        词向量和位置编码拼接并输出
        :param emb: 词向量序列（FloatTensor），``(seq_len, batch_size, self.dim)``
        :param step: 如果 stepwise("seq_len=1")，则用此位置的编码
        :return: 词向量和位置编码的拼接
        """
        # emb = emb * math.sqrt(self.dim)
        if step is None:
            # emb = emb + self.pe[:emb.size(0)]  # 拼接词向量和位置编码
            emb = emb + self.pe[:emb.size(1)].squeeze().unsqueeze(0) / math.sqrt(self.dim)
            # emb = emb + self.proj(self.pe[:emb.size(1)].squeeze().unsqueeze(0))
        else:
            if virtual=='avg':
                emb = emb + self.pe[step].squeeze().unsqueeze(0)/ math.sqrt(self.dim)
            elif virtual=='pos0':
                emb = emb + self.pe[0]/ math.sqrt(self.dim)
                # emb = emb0
            elif virtual=='node_reverse':
                # emb = emb + self.pe[step.squeeze()]/ math.sqrt(self.dim)
                for i in range(len(step)):
                    # emb[i][:step[i]] = emb[i][:step[i]] + torch.flip(self.pe[:emb.size(1)].squeeze() / math.sqrt(self.dim), [1])[:step[i]]
                    emb[i][:step[i]] = emb[i][:step[i]] + torch.flip(self.pe[1:step[i]+1].squeeze() / math.sqrt(self.dim), [1])
                # emb = emb + torch.flip(self.pe[:emb.size(1)+1].squeeze().unsqueeze(0) / math.sqrt(self.dim), [1])[:,1:]
            elif virtual=='node_reverse_woemb':
                emb=emb[:,:,:self.dim]
                for i in range(len(step)):
                    emb[i][:step[i]] = torch.flip(self.pe[1:step[i]+1].squeeze() / math.sqrt(self.dim), [1])
            else:
                emb = emb + self.pe[step.squeeze()]/ math.sqrt(self.dim)
        emb = self.drop_out(emb)
        return emb


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(d_model, max_len))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x, step, virtual):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        if virtual=='node_reverse':
            for i in range(len(step)):
                x[i][:step[i]] = x[i][:step[i]] + self.pe[1:step[i]+1].squeeze()
                # x[i][:step[i]] = x[i][:step[i]] + self.pe[:step[i]].squeeze()
        elif virtual=='pos0':
            x = x + self.pe[0]
        elif virtual=='last':
            x = x+ self.pe[step]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    if len(v1.size())==3:
        return torch.cat([v1, v2], 2)
    else:
        return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x

class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim, padding_idx=0):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        if len(user_embed.size())==3:
            x = self.fuse_embed(torch.cat((user_embed, poi_embed), 2))
        elif len(user_embed.size())==2:
            x = self.fuse_embed(torch.cat((user_embed, poi_embed), 1))
        else:
            x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
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

class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e
    
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

def gradtail_loss(grads_w, sig, loss_sample,param,alpha=0.25,p=0.):

    def fsigmoid(x):
        A=1.0
        B=0.1
        return  1.0 + (A/(1.0-np.exp(-B*x)))

    theta=[]
    grads=[]
    # for ls in loss_sample:
    loss_sample.backward(retain_graph=True)
    grads.append(param.grad.to("cpu").numpy())
    d = torch.dot(F.normalize(param.grad.to("cpu").view(-1),p=2, dim=0),F.normalize(grads_w.to("cpu").view(-1),p=2, dim=0))
    theta.append(d.numpy())

    Esig = np.mean(np.abs(theta))
    Egrad= np.mean(grads,axis=0)

    sig = alpha * sig + (1. - alpha) * Esig
    grads_w = alpha * grads_w.to("cpu") + (1. - alpha) * Egrad

    q=fsigmoid(np.abs((np.array(theta)+1e-8)/(sig+1e-8) - p))
    loss=(q.to("cuda")*loss_sample).sum()
    return loss,grads_w.to("cuda")

def GradientTailLoss(inputs, targets, alpha=0.25, beta=1, k=1):
    one_hot = torch.zeros_like(inputs).to("cuda")
    one_hot.scatter_(1, targets[:len(inputs)].view(-1, 1), 1)
    prob = torch.sigmoid(inputs)
    loss = - alpha * (1 - prob) ** k * one_hot * torch.log(prob) - (1 - one_hot) * beta * prob ** k * torch.log(1 - prob)
    return loss.mean()

class FC(nn.Module):
    def __init__(self, in_size, out_size):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)
        
    def forward(self, x):
        x = self.fc(x)
        for i in range(x.size(0)):
            x[i]=self.bn(x[i])
        return x


pos_size=23
tim_dim=512
node_dim=2000 #nodesize+1


def freedman_diaconis_bins(x, return_bins=False):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    binsize = 2 * iqr * np.power(len(x), -1/3)
    bins = np.ceil((np.max(x) - np.min(x)) / binsize)
    if return_bins:
        return int(bins), np.histogram(x, int(bins))[1]
    else:
        return int(bins)


class Graphormer(pl.LightningModule):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        weight_decay,
        ffn_dim,
        dataset_name,
        warmup_updates,
        tot_updates,
        peak_lr,
        end_lr,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        flag=False,
        flag_m=3,
        flag_step_size=1e-3,
        flag_mag=1e-3,
        lr_step=2
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_virtual_tokens = 1

        self.num_heads = num_heads
        
        if dataset_name=="toyota":
            self.atom_encoder = nn.Embedding(16460, hidden_dim, padding_idx=0) #toyota:8000,5500
            self.drop_out=0.1
            self.pos_embed = PositionalEncoding(hidden_dim, self.drop_out)
            self.edge_encoder = nn.Embedding(128, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == "multi_hop":
                self.edge_dis_encoder = nn.Embedding(128 * num_heads * num_heads, 1)
            self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(128, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(128, hidden_dim, padding_idx=0)
            
        elif dataset_name=="foursquare":
            self.atom_encoder = nn.Embedding(13963, hidden_dim, padding_idx=0) #foursquare_deep 10495, foursquare_nyc 13963, tky 21398, global 69008
            self.drop_out=0.1
            self.edge_encoder = nn.Embedding(128, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == "multi_hop":
                self.edge_dis_encoder = nn.Embedding(128 * num_heads * num_heads, 1)
            self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(128, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(128, hidden_dim, padding_idx=0)
            self.pos_embed= LearnablePositionalEncoding(node_dim, hidden_dim)
            self.time_encoder = nn.Embedding(169, tim_dim, padding_idx=0)
            
        elif dataset_name=="gowalla_7day" or dataset_name=="gowalla_nevda":
            self.drop_out=0.1
            self.edge_encoder = nn.Embedding(128, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == "multi_hop":
                self.edge_dis_encoder = nn.Embedding(128 * num_heads * num_heads, 1)
            self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)               
            # self.time_encoder = nn.Embedding(169, tim_dim)
            
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if dataset_name == "gowalla_nevda":
                path = "../dataset/gowalla_nevda/raw/"
            elif dataset_name == "gowalla_7day": 
                path = "../dataset/gowalla_7day/raw/"

            #category module
            #-----------------------
            self.raw_C = pd.read_csv(path+"Graph_cat.csv").to_numpy()
            self.C_A = calculate_laplacian_matrix(self.raw_C, mat_type='hat_rw_normd_lap_mat')
            self.C_A = torch.from_numpy(self.C_A)
            # self.D_A = self.D_A.to(device=device, dtype=torch.float)
            self.C_A = self.C_A.to(dtype=torch.float)
            #-----------------------

            #distance module
            #-----------------------
            self.raw_D = pd.read_csv(path+"Graph_dist.csv").to_numpy()
            self.D_A = calculate_laplacian_matrix(self.raw_D, mat_type='hat_rw_normd_lap_mat')
            self.D_A = torch.from_numpy(self.D_A)
            # self.D_A = self.D_A.to(device=device, dtype=torch.float)
            self.D_A = self.D_A.to(dtype=torch.float)
            #-----------------------

            self.raw_A=pd.read_csv(path+"Graph_adj.csv").to_numpy()
            # self.raw_A = np.loadtxt(path+"Graph_adj.csv", delimiter=',')
            self.raw_X=pd.read_csv(path+"Graph_poi.csv").to_numpy()
            self.nodes_df=pd.read_csv(path+"Graph_poi.csv")
            # self.atom_encoder = nn.Embedding(len(self.raw_X)+1, hidden_dim, padding_idx=0)

            self.A = calculate_laplacian_matrix(self.raw_A, mat_type='hat_rw_normd_lap_mat')
            self.A = torch.from_numpy(self.A)
            # self.A = self.A.to(device=device, dtype=torch.float)
            self.A = self.A.to(dtype=torch.float)
            one_hot_encoder = OneHotEncoder()
            cat_list = list(self.raw_X[:, 4])
            one_hot_encoder.fit(list(map(lambda x: [x], cat_list))) #他把类别做成了独热向量
            one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
            num_cats = one_hot_rlt.shape[-1]
            # num_cats=len(set(cat_list))
            num_pois = self.raw_X.shape[0]
            X = np.zeros((num_pois, 3 + num_cats), dtype=np.float32)
            X[:, 0] = self.raw_X[:, 1]
            X[:, 1:num_cats + 1] = one_hot_rlt
            X[:, num_cats + 1] = self.raw_X[:, 2]
            X[:, num_cats + 2] = self.raw_X[:, 3]
            # X[:, num_cats + 3] = self.raw_X[:, 1]

            # self.X=torch.FloatTensor(X).to(device) 
            self.X=torch.FloatTensor(X)
            num_cat_list=[]
            for i in range(num_cats):
                num_cat_list.append(i+1)
            one_hot_rlt_cat = one_hot_encoder.transform(list(map(lambda x: [x], num_cat_list))).toarray()
            self.C_X=torch.FloatTensor(one_hot_rlt_cat) 

            self.gcn_nfeat = self.X.shape[1]
            self.gcn_nhid=[16, 64]
            self.gcn_poi_embed_dim=hidden_dim
            self.gcn_dropout=0.3
            self.gcn_cat_nfeat = self.C_X.shape[1]
            self.gcn_cat_dropout=0.1

            self.poi_embed_model = GCN(ninput=self.gcn_nfeat,
                        nhid=self.gcn_nhid,
                        noutput=self.gcn_poi_embed_dim,
                        dropout=self.gcn_dropout)
            self.fuse_embed = nn.Linear(2*hidden_dim, hidden_dim)
            # self.leaky_relu = nn.LeakyReLU(0.2)

            self.user_embed_dim = hidden_dim
            self.poi_embed_dim = hidden_dim
            self.time_embed_dim = 32
            self.cat_embed_dim = 32
            #NYC From GETNext and Houston 1057 TKY 2261 Gowalla 1972 Gowalla nevda 1080/7day 937
            if dataset_name == "gowalla_nevda":
                self.num_users = 1080  
            elif dataset_name == "gowalla_7day": 
                self.num_users = 937

            self.poi_distance_model=GCN(ninput=self.gcn_nfeat,
                        nhid=self.gcn_nhid,
                        noutput=self.gcn_poi_embed_dim,
                        dropout=self.gcn_dropout)

            self.poi_cat_model=GCN(ninput=self.gcn_cat_nfeat,
                        nhid=self.gcn_nhid,
                        noutput=self.cat_embed_dim,
                        dropout=self.gcn_cat_dropout) 
            
            self.user_embed_model = UserEmbeddings(self.num_users, self.user_embed_dim).to(self.device)

            # %% Model3: Time Model
            # self.time_embed_model = Time2Vec('sin', out_dim=self.time_embed_dim).to(device)
            # self.time_embed_model = Time2Vec('sin', out_dim=self.time_embed_dim)
            self.time_embed_model_48 = nn.Embedding(48, self.time_embed_dim, padding_idx=0).to(device)

            # %% Model4: Category embedding model
            # self.cat_embed_model = CategoryEmbeddings(num_cats+1, self.cat_embed_dim).to(device)
            self.cat_embed_model = CategoryEmbeddings(num_cats+1, self.cat_embed_dim)
            self.cat_target=0

            self.cat_decoder = nn.Linear(hidden_dim*2+ self.time_embed_dim+ self.cat_embed_dim,  num_cats+1).to(device)
            self.embed_fuse_model1 = FuseEmbeddings(self.user_embed_dim, self.poi_embed_dim).to(self.device)
            self.embed_fuse_model2 = FuseEmbeddings(self.poi_embed_dim, self.time_embed_dim).to(self.device)
            self.embed_fuse_model3 = FuseEmbeddings(self.user_embed_dim, self.poi_embed_dim+self.time_embed_dim+self.cat_embed_dim).to(self.device)
            self.embed_fuse_model4 = FuseEmbeddings(self.poi_embed_dim+self.time_embed_dim, self.cat_embed_dim).to(self.device)

            # self.node_attn_model = NodeAttnMap(in_features=self.X.shape[1], nhid=hidden_dim, use_mask=False).to(device)

            self.pos_embed= LearnablePositionalEncoding(node_dim, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim)
            self.in_degree_encoder = nn.Embedding(128, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(128, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)

            # self.pos_embed= LearnablePositionalEncoding(node_dim, hidden_dim)
            # self.in_degree_encoder = nn.Embedding(128, hidden_dim, padding_idx=0)
            # self.out_degree_encoder = nn.Embedding(128, hidden_dim, padding_idx=0)
            
            if dataset_name == "gowalla_nevda":
                self.fre_embed_model = nn.Embedding(self.nodes_df['check_freq'].max()+1, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)
            elif dataset_name == "gowalla_7day": 
                self.fre_embed_model = nn.Embedding(self.nodes_df['checkin_cnt'].max()+1, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)

            # self.POI_norm1=nn.Linear(hidden_dim, hidden_dim).to(device)
            # self.fuseemb_norm=FC(self.time_embed_dim+ self.cat_embed_dim, self.time_embed_dim+ self.cat_embed_dim)
            self.output_dropout = nn.Dropout(intput_dropout_rate)

            poi_distance_matrix = pickle.load(open('../dataset/poi_data/gowalla_distance.pkl', 'rb'), encoding='iso-8859-1')
            distance_matrix_row=np.delete(poi_distance_matrix,0,axis=0)
            # np.delete(distance_matrix,0,axis=1)
            distance_matrix=np.delete(distance_matrix_row,0,axis=1)
            distances_adjusted = distance_matrix - distance_matrix.min()
            num_bins = freedman_diaconis_bins(distances_adjusted)

            self.poi_pos_encoder = nn.Embedding(num_bins, num_heads, padding_idx=0)
            
        elif dataset_name=="foursquaregraph":
            # self.atom_encoder = nn.Embedding(7856, hidden_dim, padding_idx=0) #NYC GETNext 4980, myself 5117, TKY 7856
            self.drop_out=0.1
            self.edge_encoder = nn.Embedding(128, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == "multi_hop":
                self.edge_dis_encoder = nn.Embedding(128 * num_heads * num_heads, 1)
            self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
            self.time_encoder = nn.Embedding(48, tim_dim)
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            path = "../dataset/foursquaregraph/raw/"
            self.raw_C = pd.read_csv(path+"Graph_cat.csv").to_numpy()
            self.C_A = calculate_laplacian_matrix(self.raw_C, mat_type='hat_rw_normd_lap_mat')
            self.C_A = torch.from_numpy(self.C_A)
            self.C_A = self.C_A.to(dtype=torch.float)
            #-----------------------

            #distance module
            #-----------------------
            self.raw_D = pd.read_csv(path+"Graph_dist.csv").to_numpy()
            self.D_A = calculate_laplacian_matrix(self.raw_D, mat_type='hat_rw_normd_lap_mat')
            self.D_A = torch.from_numpy(self.D_A)
            self.D_A = self.D_A.to(dtype=torch.float)
            #-----------------------

            self.raw_A=pd.read_csv(path+"Graph_adj.csv").to_numpy()
            # self.raw_A = np.loadtxt(path+"Graph_adj.csv", delimiter=',')
            self.raw_X=pd.read_csv(path+"Graph_poi.csv").to_numpy()
            self.nodes_df=pd.read_csv(path+"Graph_poi.csv")
            self.atom_encoder = nn.Embedding(len(self.raw_X), hidden_dim, padding_idx=0)

            self.A = calculate_laplacian_matrix(self.raw_A, mat_type='hat_rw_normd_lap_mat')
            self.A = torch.from_numpy(self.A)
            self.A = self.A.to(dtype=torch.float)
            one_hot_encoder = OneHotEncoder()
            cat_list = list(self.raw_X[:, 4])
            one_hot_encoder.fit(list(map(lambda x: [x], cat_list))) #他把类别做成了独热向量
            one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
            num_cats = one_hot_rlt.shape[-1]
            num_pois = self.raw_X.shape[0]
            X = np.zeros((num_pois, 3 + num_cats), dtype=np.float32)
            X[:, 0] = self.raw_X[:, 1]
            X[:, 1:num_cats + 1] = one_hot_rlt
            X[:, num_cats + 1] = self.raw_X[:, 2]
            X[:, num_cats + 2] = self.raw_X[:, 3]

            self.X=torch.FloatTensor(X)
            num_cat_list=[]
            for i in range(num_cats):
                num_cat_list.append(i+1)
            one_hot_rlt_cat = one_hot_encoder.transform(list(map(lambda x: [x], num_cat_list))).toarray()
            self.C_X=torch.FloatTensor(one_hot_rlt_cat) 

            self.gcn_nfeat = self.X.shape[1]
            self.gcn_nhid=[16, 64]
            self.gcn_poi_embed_dim=hidden_dim
            self.gcn_dropout=0.3
            self.gcn_cat_nfeat = self.C_X.shape[1]
            self.gcn_cat_dropout=0.1
            
            self.poi_embed_model = GCN(ninput=self.gcn_nfeat,
                        nhid=self.gcn_nhid,
                        noutput=self.gcn_poi_embed_dim,
                        dropout=self.gcn_dropout)
            self.fuse_embed = nn.Linear(2*hidden_dim, hidden_dim)
            self.leaky_relu = nn.LeakyReLU(0.2)

            self.user_embed_dim = hidden_dim
            self.poi_embed_dim = hidden_dim
            self.time_embed_dim = 32
            self.cat_embed_dim = 32
            self.num_users = 1080  #NYC From GETNext and Houston 1057 TKY 2261 Gowalla 1972 Gowalla nevda 1080/7day 937

            self.poi_distance_model=GCN(ninput=self.gcn_nfeat,
                        nhid=self.gcn_nhid,
                        noutput=self.gcn_poi_embed_dim,
                        dropout=self.gcn_dropout)

            self.poi_cat_model=GCN(ninput=self.gcn_cat_nfeat,
                        nhid=self.gcn_nhid,
                        noutput=self.cat_embed_dim,
                        dropout=self.gcn_cat_dropout) 
            
            self.user_embed_model = UserEmbeddings(self.num_users, self.user_embed_dim).to(self.device)

            # %% Model3: Time Model
            # self.time_embed_model = Time2Vec('sin', out_dim=self.time_embed_dim).to(device)
            # self.time_embed_model = Time2Vec('sin', out_dim=self.time_embed_dim)
            self.time_embed_model_48 = nn.Embedding(48+1, self.time_embed_dim, padding_idx=0).to(device)

            # %% Model4: Category embedding model
            # self.cat_embed_model = CategoryEmbeddings(num_cats+1, self.cat_embed_dim).to(device)
            self.cat_embed_model = CategoryEmbeddings(num_cats, self.cat_embed_dim)
            self.cat_target=0

            self.cat_decoder = nn.Linear(hidden_dim*2+ self.time_embed_dim+ self.cat_embed_dim,  num_cats).to(device)

            self.embed_fuse_model1 = FuseEmbeddings(self.user_embed_dim, self.poi_embed_dim).to(self.device)
            self.embed_fuse_model2 = FuseEmbeddings(self.poi_embed_dim, self.time_embed_dim).to(self.device)
            self.embed_fuse_model3 = FuseEmbeddings(self.user_embed_dim, self.poi_embed_dim+self.time_embed_dim+self.cat_embed_dim).to(self.device)
            self.embed_fuse_model4 = FuseEmbeddings(self.poi_embed_dim+self.time_embed_dim, self.cat_embed_dim).to(self.device)

            # self.node_attn_model = NodeAttnMap(in_features=self.X.shape[1], nhid=hidden_dim, use_mask=False).to(device)
            self.pos_embed= LearnablePositionalEncoding(node_dim, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim)
            self.in_degree_encoder = nn.Embedding(128, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(128, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)
            
            self.fre_embed_model = nn.Embedding(self.nodes_df['check_freq'].max()+1, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)

            # self.POI_norm1=nn.Linear(hidden_dim, hidden_dim).to(device)
            # self.fuseemb_norm=FC(self.time_embed_dim+ self.cat_embed_dim, self.time_embed_dim+ self.cat_embed_dim)
            self.output_dropout = nn.Dropout(intput_dropout_rate)

            poi_distance_matrix = pickle.load(open('../dataset/poi_data/tky_distance.pkl', 'rb'), encoding='iso-8859-1')
            distance_matrix=np.delete(poi_distance_matrix,0,axis=0)
            np.delete(distance_matrix,0,axis=1)
            distances_adjusted = distance_matrix - distance_matrix.min()
            num_bins = freedman_diaconis_bins(distances_adjusted)

            self.poi_pos_encoder = nn.Embedding(num_bins, num_heads, padding_idx=0)
        
        elif dataset_name=="toyotagraph":
            self.drop_out=0.1
            self.edge_encoder = nn.Embedding(128, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == "multi_hop":
                self.edge_dis_encoder = nn.Embedding(128 * num_heads * num_heads, 1)
            self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)               
            
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            path="../dataset/toyotagraph/raw/"

            #category module
            #-----------------------
            self.raw_C = pd.read_csv(path+"Graph_cat.csv").to_numpy()
            self.C_A = calculate_laplacian_matrix(self.raw_C, mat_type='hat_rw_normd_lap_mat')
            self.C_A = torch.from_numpy(self.C_A)
            # self.D_A = self.D_A.to(device=device, dtype=torch.float)
            self.C_A = self.C_A.to(dtype=torch.float)
            #-----------------------

            #distance module
            #-----------------------
            self.raw_D = pd.read_csv(path+"Graph_dist.csv").to_numpy()
            self.D_A = calculate_laplacian_matrix(self.raw_D, mat_type='hat_rw_normd_lap_mat')
            self.D_A = torch.from_numpy(self.D_A)
            # self.D_A = self.D_A.to(device=device, dtype=torch.float)
            self.D_A = self.D_A.to(dtype=torch.float)
            #-----------------------

            self.raw_A=pd.read_csv(path+"Graph_adj.csv").to_numpy()
            # self.raw_A = np.loadtxt(path+"Graph_adj.csv", delimiter=',')
            self.raw_X=pd.read_csv(path+"Graph_poi.csv").to_numpy()
            self.nodes_df=pd.read_csv(path+"Graph_poi.csv")
            # self.atom_encoder = nn.Embedding(len(self.raw_X)+1, hidden_dim, padding_idx=0)

            self.A = calculate_laplacian_matrix(self.raw_A, mat_type='hat_rw_normd_lap_mat')
            self.A = torch.from_numpy(self.A)
            # self.A = self.A.to(device=device, dtype=torch.float)
            self.A = self.A.to(dtype=torch.float)
            one_hot_encoder = OneHotEncoder()
            cat_list = list(self.raw_X[:, 4])
            one_hot_encoder.fit(list(map(lambda x: [x], cat_list))) #他把类别做成了独热向量
            one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
            num_cats = one_hot_rlt.shape[-1]
            # num_cats=len(set(cat_list))
            num_pois = self.raw_X.shape[0]
            X = np.zeros((num_pois, 3 + num_cats), dtype=np.float32)
            X[:, 0] = self.raw_X[:, 1]
            X[:, 1:num_cats + 1] = one_hot_rlt
            X[:, num_cats + 1] = self.raw_X[:, 2]
            X[:, num_cats + 2] = self.raw_X[:, 3]
            # X[:, num_cats + 3] = self.raw_X[:, 1]

            # self.X=torch.FloatTensor(X).to(device) 
            self.X=torch.FloatTensor(X)
            num_cat_list=[]
            for i in range(num_cats):
                num_cat_list.append(i+1)
            one_hot_rlt_cat = one_hot_encoder.transform(list(map(lambda x: [x], num_cat_list))).toarray()
            self.C_X=torch.FloatTensor(one_hot_rlt_cat) 

            self.gcn_nfeat = self.X.shape[1]
            self.gcn_nhid=[16, 64]
            self.gcn_poi_embed_dim=hidden_dim
            self.gcn_dropout=0.3
            self.gcn_cat_nfeat = self.C_X.shape[1]
            self.gcn_cat_dropout=0.1

            self.poi_embed_model = GCN(ninput=self.gcn_nfeat,
                        nhid=self.gcn_nhid,
                        noutput=self.gcn_poi_embed_dim,
                        dropout=self.gcn_dropout)
            self.fuse_embed = nn.Linear(2*hidden_dim, hidden_dim)
            self.leaky_relu = nn.LeakyReLU(0.2)

            self.user_embed_dim = hidden_dim
            self.poi_embed_dim = hidden_dim
            self.time_embed_dim = 32
            self.cat_embed_dim = 32
            self.num_users = 995  #NYC From GETNext and Houston 1057 TKY 2261 Gowalla 1972 Gowalla nevda 1080/7day 937

            self.poi_distance_model=GCN(ninput=self.gcn_nfeat,
                        nhid=self.gcn_nhid,
                        noutput=self.gcn_poi_embed_dim,
                        dropout=self.gcn_dropout)

            self.poi_cat_model=GCN(ninput=self.gcn_cat_nfeat,
                        nhid=self.gcn_nhid,
                        noutput=self.cat_embed_dim,
                        dropout=self.gcn_cat_dropout) 
            
            self.user_embed_model = UserEmbeddings(self.num_users+1, self.user_embed_dim).to(self.device)

            # %% Model3: Time Model
            # self.time_embed_model = Time2Vec('sin', out_dim=self.time_embed_dim).to(device)
            # self.time_embed_model = Time2Vec('sin', out_dim=self.time_embed_dim)
            self.time_embed_model_48 = nn.Embedding(48, self.time_embed_dim).to(device)

            # %% Model4: Category embedding model
            # self.cat_embed_model = CategoryEmbeddings(num_cats+1, self.cat_embed_dim).to(device)
            # self.cat_embed_model = CategoryEmbeddings(num_cats, self.cat_embed_dim)
            self.cat_target=0

            self.cat_decoder = nn.Linear(hidden_dim*2+ self.time_embed_dim+ self.cat_embed_dim,  num_cats).to(device)
            self.embed_fuse_model1 = FuseEmbeddings(self.user_embed_dim, self.poi_embed_dim).to(self.device)
            self.embed_fuse_model2 = FuseEmbeddings(self.poi_embed_dim, self.time_embed_dim).to(self.device)
            self.embed_fuse_model3 = FuseEmbeddings(self.user_embed_dim, self.poi_embed_dim+self.time_embed_dim+self.cat_embed_dim).to(self.device)
            self.embed_fuse_model4 = FuseEmbeddings(self.poi_embed_dim+self.time_embed_dim, self.cat_embed_dim).to(self.device)

            # self.node_attn_model = NodeAttnMap(in_features=self.X.shape[1], nhid=hidden_dim, use_mask=False).to(device)

            self.pos_embed= LearnablePositionalEncoding(node_dim, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim)
            self.in_degree_encoder = nn.Embedding(128, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(128, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)
            
            self.fre_embed_model = nn.Embedding(self.nodes_df['check_freq'].max()+1, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, padding_idx=0)

            # self.POI_norm1=nn.Linear(hidden_dim, hidden_dim).to(device)
            # self.fuseemb_norm=FC(self.time_embed_dim+ self.cat_embed_dim, self.time_embed_dim+ self.cat_embed_dim)
            self.output_dropout = nn.Dropout(intput_dropout_rate)

            poi_distance_matrix = pickle.load(open('../dataset/poi_data/toyota_distance.pkl', 'rb'), encoding='iso-8859-1')
            distance_matrix_row=np.delete(poi_distance_matrix,0,axis=0)
            distance_matrix = np.delete(distance_matrix_row,0,axis=1)
            distances_adjusted = distance_matrix - distance_matrix.min()
            num_bins = freedman_diaconis_bins(distances_adjusted)

            self.poi_pos_encoder = nn.Embedding(num_bins, num_heads, padding_idx=0)

        else:
            self.atom_encoder = nn.Embedding(512 * 9 + 1, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(512 * 3 + 1, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == "multi_hop":
                self.edge_dis_encoder = nn.Embedding(128 * num_heads * num_heads, 1)
            self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [
            EncoderLayer(
                hidden_dim+ self.time_embed_dim+ self.cat_embed_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads
                # hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders) #transformer块
        self.final_ln = nn.LayerNorm(hidden_dim*2+ self.time_embed_dim+ self.cat_embed_dim)
        # self.final_ln = nn.LayerNorm(hidden_dim)

        if dataset_name=="toyota":
            self.out_proj = nn.Linear(hidden_dim, 16460) #toyota:8000,5500
        elif dataset_name=='foursquare':
            self.out_proj = nn.Linear(hidden_dim, 13963) #foursquare:10495,foursquare_nyc13963
            # self.proj_matrix = nn.Linear(hidden_dim*2, hidden_dim) 
        elif dataset_name=='foursquaregraph':
            self.out_proj = nn.Linear(hidden_dim*2+ self.time_embed_dim+ self.cat_embed_dim, len(self.raw_X))
            # self.out_proj = nn.Linear(hidden_dim, len(self.raw_X))
            # self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=0.3, inplace=False)
            self.ELU = torch.nn.ELU()
        elif dataset_name == 'gowalla_7day' or dataset_name == 'gowalla_nevda':
            self.out_proj = nn.Linear(hidden_dim*2+ self.time_embed_dim+ self.cat_embed_dim, len(self.raw_X)+1) # An error happens when +1 is removed
            # self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=0.3, inplace=False)
            self.ELU = torch.nn.ELU()
        elif dataset_name=='toyotagraph':  # ADDED toyotagraph from graphormer_new_cuda2
            self.out_proj = nn.Linear(hidden_dim*2+ self.time_embed_dim+ self.cat_embed_dim, len(self.raw_X)+1)
            # self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=0.3, inplace=False)
            self.ELU = torch.nn.ELU()
        else:
            self.downstream_out_proj = nn.Linear(
                hidden_dim, get_dataset(dataset_name)["num_class"]
            )
        self.graph_token = nn.Embedding(self.num_virtual_tokens, hidden_dim+ self.time_embed_dim+ self.cat_embed_dim)
        # self.graph_token = nn.Embedding(self.num_virtual_tokens, hidden_dim)

        self.graph_token_virtual_distance = nn.Embedding(self.num_virtual_tokens, num_heads)

        self.evaluator = get_dataset(dataset_name)["evaluator"]
        self.metric = get_dataset(dataset_name)["metric"]
        self.loss_fn = get_dataset(dataset_name)["loss_fn"]
        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist
        self.lr_step=lr_step

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.automatic_optimization = not self.flag
        # self.apply(lambda module: init_bert_params(module, n_layers=n_layers))
        self.attn = Attn('dot', self.hidden_dim)
        self.indx=0
        self.poi_emb=0

        # Categroy to embedding

        self.poi_idx2cat_idx_dict = {}
        for i, row in self.nodes_df.iterrows():
            self.poi_idx2cat_idx_dict[row['POI ID']] =[row['cat']]

        # Popularity to embedding

        self.poi_idx2freq_idx_dict = {}
        for i, row in self.nodes_df.iterrows():
            if dataset_name == 'gowalla_7day':
                self.poi_idx2freq_idx_dict[row['POI ID']] = [row['checkin_cnt']]
            else:
                self.poi_idx2freq_idx_dict[row['POI ID']] = [row['check_freq']]

        # self.loss_POI = nn.CrossEntropyLoss(ignore_index=0)
        # self.loss_focal=FocalLoss()
       

    def forward(self, batched_data, perturb=None):
                
        attn_bias, rel_pos, x = (
            batched_data.attn_bias,
            batched_data.rel_pos,
            batched_data.x,
        )
        in_degree, out_degree = batched_data.in_degree, batched_data.out_degree
        edge_input, attn_edge_type = (
            batched_data.edge_input,
            batched_data.attn_edge_type,
        )
        self.cat_target=copy.deepcopy(batched_data.y)
        poi_pos=batched_data.poi_pos
        # time=batched_data.time
        # time_normal=(batched_data.time_normal*48+1)/48
        time_normal=batched_data.time_normal
        user=batched_data.user
        cat=batched_data.cat

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # rel pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        rel_poi_bias= self.poi_pos_encoder(poi_pos).permute(0, 3, 1, 2)


        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, self.num_virtual_tokens:, self.num_virtual_tokens:] = (
            graph_attn_bias[:, :, self.num_virtual_tokens:, self.num_virtual_tokens:]
            + rel_pos_bias+rel_poi_bias
        )
        # reset rel pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, self.num_virtual_tokens).unsqueeze(-2)  # [1,8,2]
        # self.graph_token_virtual_distance.weight.view(1, self.num_heads, self.num_virtual_tokens).unsqueeze(-2)
        graph_attn_bias[:, :, self.num_virtual_tokens:, :self.num_virtual_tokens] = (
            graph_attn_bias[:, :, self.num_virtual_tokens:, :self.num_virtual_tokens]
            + t  # [256, 8, 35, 2]
        )

        # edge feature
        if self.edge_type == "multi_hop":
            rel_pos_ = rel_pos.clone()
            rel_pos_[rel_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            rel_pos_ = torch.where(rel_pos_ > 1, rel_pos_ - 1, rel_pos_)
            if self.multi_hop_max_dist > 0:
                rel_pos_ = rel_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]             # [n_graph, n_node, n_node, max_dist, n_head]
            # edge_input = self.edge_encoder(edge_input).mean(-2)
            temp_edge=self.edge_encoder(edge_input)
            edge_input=torch.cat([temp_edge[j].mean(-2).unsqueeze(0).half() for j in range(len(edge_input))],dim=0).float()
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            try:
                edge_input_flat = torch.bmm(
                edge_input_flat.half(),
                self.edge_dis_encoder.weight.half().reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
                )
            except:
                torch.cuda.empty_cache()
                edge_input_flat = torch.bmm(
                edge_input_flat.half(),
                self.edge_dis_encoder.weight.half().reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
                )
            edge_input_flat=edge_input_flat.float()
            #看起来是为了toyota,下面这个好像是我写的
            # edge_input = edge_input_flat.reshape(
            #     max_dist, n_graph, int(math.pow(edge_input_flat.size(1),0.5)), int(math.pow(edge_input_flat.size(1),0.5)), self.num_heads
            # ).permute(1, 2, 3, 0, 4)
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (rel_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, self.num_virtual_tokens:, self.num_virtual_tokens:] = (
            graph_attn_bias[:, :, self.num_virtual_tokens:, self.num_virtual_tokens:] + edge_input
        )
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        # print(x.size())
        # node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        # cat_embedding = self.cat_embed_model(cat).sum(dim=-2)
        node_feature = torch.zeros(x.size(0),x.size(1),self.hidden_dim).to(self.device)
        time_embedding = torch.zeros(x.size(0),x.size(1),self.time_embed_dim).to(self.device)

        mask = x == 0
        mask = ~mask
        indx = mask.sum(dim=-2)
        self.indx=mask.sum(dim=-2)

        if self.flag and perturb is not None:
            node_feature += perturb
        

        #------------------------GETNext---------------------------
        # self.global_poiemb=self.poi_embed_model(self.X.to(self.device), self.A.to(self.device)).to(self.device)
        self.global_poidistemb=self.poi_distance_model(self.X.to(self.device), self.D_A.to(self.device)).to(self.device)
        self.global_catemb=self.poi_cat_model(self.C_X.to(self.device), self.C_A.to(self.device)).to(self.device)
             
        user_embedding = self.user_embed_model(user-1).to(self.device)
        user_embedding = torch.squeeze(user_embedding)

        cat_embedding=torch.zeros([x.size(0),x.size(1),self.cat_embed_dim]).to(self.device)
        poi_freq=torch.zeros([x.size(0),x.size(1),1]).long().to(self.device)

        # temp_node_features=torch.cat([node_feature, node_feature], 2)   


        node_features= torch.zeros([x.size(0),x.size(1),self.hidden_dim+self.time_embed_dim+self.cat_embed_dim]).to(self.device)
        # node_feature= torch.zeros([x.size(0),x.size(1),self.hidden_dim])
        # fused_embedding2 = torch.cat([time_embedding, cat_embedding], 2)
        fused_embedding2 = torch.cat([node_feature, time_embedding], 2)
        # fused_embedding2 = torch.cat([node_feature, time_embedding], 2)
        # freq_embedding= torch.zeros([x.size(0),x.size(1),self.hidden_dim+self.time_embed_dim+self.cat_embed_dim])



        for p in range(x.size(0)):
            length=indx[p][0].item()
            cat_embedding[p][:length]=self.global_catemb[torch.LongTensor([self.poi_idx2cat_idx_dict[int(x[p][q])][0]-1 for q in range(length)])]
            # poi_freq[p][:length]=torch.LongTensor([self.poi_idx2freq_idx_dict[int(x[p][q])] for q in range(length)])
            # time_embedding[p][:length]=self.time_embed_model(time_normal[p][:length])
            time_embedding[p][:length]=self.time_embed_model_48((time_normal[p][:length]*48).long()).squeeze(1)
            # node_feature[p][:length]=((self.global_poiemb[x[p][:length]-1]+self.global_poidistemb[x[p][:length]-1])/2).squeeze(1)
            node_feature[p][:length]=(self.global_poidistemb[x[p][:length]-1]).squeeze(1)
            self.cat_target[p]=torch.LongTensor([self.poi_idx2cat_idx_dict[int(self.cat_target[p])][0]-1]).to(self.device)
            # node_features[p][:length]=((self.global_poiemb[x[p][:length]-1]+self.global_poidistemb[x[p][:length]-1])/2).squeeze(1)
            # fused_embedding2[p][:length] = self.embed_fuse_model2(node_feature[p][:length], time_embedding[p][:length])
            fused_embedding2[p][:length] = self.embed_fuse_model2(node_feature[p][:length], time_embedding[p][:length])
            node_features[p][:length] = self.embed_fuse_model4(fused_embedding2[p][:length], cat_embedding[p][:length])


        # freq_embedding = self.fre_embed_model(poi_freq).squeeze(2)
        # node_feature = (
        #     node_feature
        #     +freq_embedding
        #     # +freq_embedding[p].squeeze(1)
        #     # +laplace_pos
        #     # pos_embedding
        #     # + time_feature
        #     + self.in_degree_encoder(in_degree)
        #     + self.out_degree_encoder(out_degree)
        # )

        # for p in range(x.size(0)):
        #     length=indx[p].item()
        #     node_features[p][:length] = self.embed_fuse_model4(node_feature[p][:length], fused_embedding2[p][:length])

        freq_embedding = self.fre_embed_model(poi_freq).squeeze(2)
        node_features = (
            node_features.to(self.device)
            +freq_embedding.to(self.device)
            # +freq_embedding[p].squeeze(1)
            # +laplace_pos
            # pos_embedding
            # + time_feature
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        # cat_embedding = self.cat_embed_model(cat.to(self.device)).to(self.device).squeeze(2)
        # cat_embedding = torch.squeeze(cat_embedding)
        

        # fused_embedding1 = self.embed_fuse_model1(user_embedding, node_features)
        # fused_embedding2 = self.embed_fuse_model2(time_embedding, cat_embedding)

        # Concat time, cat after user+poi
        # node_feature_fc = self.POI_norm1(node_feature)
        # fused_embedding2_fc= self.fuseemb_norm(fused_embedding2)
        # node_features = self.embed_fuse_model4(node_feature_fc, fused_embedding2)
        # node_features = torch.cat((temp_node_features, fused_embedding2), dim=-1)


        # node_feature = (
        #     node_features
        #     +freq_embedding
        #     # +laplace_pos
        #     # pos_embedding
        #     # + time_feature
        #     + self.in_degree_encoder(in_degree)
        #     + self.out_degree_encoder(out_degree)
        # )

        node_features = self.pos_embed(node_features.to(self.device),step=indx,virtual="node_reverse")
        

        #--------------------------------------------------

        # node_features = (
        #     node_feature
        #     # +laplace_pos
        #     # pos_embedding
        #     # + time_feature
        #     + self.in_degree_encoder(in_degree)
        #     + self.out_degree_encoder(out_degree)
        # )
        
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        # graph_token_feature += index_pos_embedding[indx]
        # graph_token_feature = self.virtual_pos_embed(graph_token_feature, step=indx,virtual='pos0')
        # graph_token_feature = (node_features.sum(dim=1)/indx).unsqueeze(1)
        graph_token_feature = self.pos_embed(graph_token_feature, step=x.size(1), virtual='pos0')
        # graph_token_feature = self.pos_embed(graph_token_feature, step=indx, virtual='last')
        graph_node_feature = torch.cat([graph_token_feature, node_features], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(
                output, graph_attn_bias, mask=None
                # output, mask=None
            )  # TODO readd mask as adj
        temp_node_features=torch.zeros(x.size(0), x.size(1), self.hidden_dim*2+self.time_embed_dim+ self.cat_embed_dim).to(self.device)
        for p in range(x.size(0)):
            # fused_list = [self.embed_fuse_model3(output[p][q], self.global_useremb[user[p]-1].squeeze(0)) for q in range(x.size(1))]
            fused_list = [self.embed_fuse_model3(output[p][q], user_embedding[p]) for q in range(x.size(1))]
            fused_embedding1 = torch.cat([fused_list[q] for q in range(len(fused_list))])
            temp_node_features[p]=fused_embedding1.reshape(x.size(1), self.hidden_dim*2+self.time_embed_dim+ self.cat_embed_dim).to(self.device)
        # output = self.final_ln(output)
        output = self.final_ln(temp_node_features)
        # output = self.LeakyReLU(output)
        output = self.ELU(output)
        # output = self.final_ln(output)
        output = self.output_dropout(output)
        # output = self.final_ln(output)

        # output part
        if self.dataset_name == "foursquare":
            # for j in range(len(output)):
                # if j==0:
                    # new_output=torch.cat([output[j,0,:].unsqueeze(0), output[j,2:,:]], dim=0).unsqueeze(0)
                # else:
                    # temp_output=torch.cat([output[j,0,:].unsqueeze(0), output[j,2:,:]], dim=0).unsqueeze(0)
                    # new_output=torch.cat([new_output,temp_output])
            # output=self.out_proj(new_output)
            # P=self.atom_encoder(torch.LongTensor(self.POI).to(self.device))
            # output=torch.mm(P,output.T.float()).T
            # for j in range(len(output)):
                # output[j]=P*output[j]
            # output = self.out_proj((torch.cat([output[:, 0, :].unsqueeze(1), output[:, 2:, :]], dim=1)))

            output = self.out_proj(output[:, 0, :]) 
            output=F.log_softmax(output)

            # attn_weights = self.attn(output[-len(batched_data.y):], output).unsqueeze(0)
            # output=attn_weights.bmm(output.unsqueeze(0)).squeeze(0)
            # output = self.out_recomm(output)
            # output=F.log_softmax(output)
        elif self.dataset_name=="toyota":
            output = self.out_proj(output[:, 0, :])
            output=F.log_softmax(output)
            
        elif self.dataset_name == "foursquaregraph": 
            cat_output=self.cat_decoder(output[:, 0, :])
            outputs=[]
            output = self.out_proj(output[:, 0, :]) 
            # output = self.LeakyReLU(output) 
            # y_pred_poi_adjusted = torch.zeros_like(output)
            # output=y_pred_poi_adjusted
            # output=F.log_softmax(output)
            outputs.append(output)
            outputs.append(cat_output)
            output=outputs
            
        elif self.dataset_name == "gowalla_7day" or self.dataset_name == "gowalla_nevda":
            cat_output=self.cat_decoder(output[:, 0, :])
            outputs=[]
            output = self.out_proj(output[:, 0, :]) 
            # output = self.LeakyReLU(output) 
            # y_pred_poi_adjusted = torch.zeros_like(output)
            # output=y_pred_poi_adjusted
            # output=F.log_softmax(output)
            outputs.append(output)
            outputs.append(cat_output)
            output=outputs
        
        # ADDED toyotagraph from graphormer_new_cuda2
        elif self.dataset_name == "toyotagraph": 
            cat_output=self.cat_decoder(output[:, 0, :])
            outputs=[]
            output = self.out_proj(output[:, 0, :]) 
            # output = self.LeakyReLU(output) 
            # y_pred_poi_adjusted = torch.zeros_like(output)
            # output=y_pred_poi_adjusted
            output=F.log_softmax(output)
            outputs.append(output)
            outputs.append(cat_output)
            output=outputs

        else:
            output = self.downstream_out_proj(output[:, 0, :])
        return output

    def training_step(self, batched_data, batch_idx):
        if self.dataset_name=="foursquare":
            y_gt = batched_data.y
            # y_gt=(torch.cat([batched_data.y.unsqueeze(1).unsqueeze(1), batched_data.x[:,1:self.indx.squeeze(0)-1]], dim=1))
            y_hat= self(batched_data)
            loss = self.loss_fn(y_hat, y_gt)
            
        elif self.dataset_name=="toyota":
            y_gt = batched_data.y.view(-1)
            y_hat= self(batched_data)
            loss = self.loss_fn(y_hat, y_gt)

        elif self.dataset_name=="foursquaregraph":
            y_gt = batched_data.y-1
            y_out = self(batched_data)
            y_cat_hat=y_out[1]
            y_hat= y_out[0]
            y_cat_gt=self.cat_target
            loss = GradientTailLoss(y_hat, y_gt, 0.2)
            
        elif self.dataset_name == "gowalla_7day" or self.dataset_name == "gowalla_nevda":
            y_gt = batched_data.y-1
            y_out = self(batched_data)
            y_cat_hat=y_out[1]
            y_hat= y_out[0]
            y_cat_gt=self.cat_target
            loss = GradientTailLoss(y_hat, y_gt, 0.2)
        
        # ADDED toyotagraph from graphormer_new_cuda2
        elif self.dataset_name=="toyotagraph":
            y_gt = batched_data.y
            y_out = self(batched_data)
            y_cat_hat=y_out[1]
            y_hat= y_out[0]
            y_cat_gt=self.cat_target
            loss1 = GradientTailLoss(y_cat_hat, y_cat_gt, 0.1)
            loss2 = self.loss_fn(y_hat, y_gt)
            loss=loss1+loss2
            
        else:
            y_hat = self(batched_data).view(-1)
            y_gt = batched_data.y.view(-1)
            loss = self.loss_fn(y_hat, y_gt)
        
        return loss
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.log_metrics({"training_loss": avg_loss}, step=self.trainer.current_epoch)

    def validation_step(self, batched_data, batch_idx):
        if self.dataset_name in ["foursquare", "toyota", "foursquaregraph", "gowalla_7day", "gowalla_nevda"]:
            y_pred = self(batched_data)
            y_true = batched_data.y-1
            # y_true=(torch.cat([batched_data.y.unsqueeze(1).unsqueeze(1), batched_data.x[:,1:]], dim=1))
            # y_pred = self(batched_data)
        else:
            y_pred = self(batched_data)
            y_true = batched_data.y
        return {
            "y_pred": y_pred,
            "y_true": y_true,
        }

    def validation_epoch_end(self, outputs):
        queue_len = len(outputs)
        users_acc = {}
        for u in range(queue_len):
            y_pred = outputs[u]["y_pred"][0]
            y_true = outputs[u]["y_true"]
            if u not in users_acc:
                users_acc[u] = [0, 0, 0]
            users_acc[u][0] += len(y_true)
            # users_acc[u][0] += (len(y_true)*y_true.size(1))
            acc,_ = get_acc(y_true, y_pred)
            # acc,_ = get_acc1(y_true, y_pred)
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
        try:
            self.log(
                "valid_" + self.metric,
                fin_acc,
                sync_dist=True,
            )
            self.logger.log_metrics({"val_acc": fin_acc}, step=self.trainer.current_epoch)
        except:
            pass

    def test_step(self, batched_data, batch_idx):
        if self.dataset_name in ["foursquare", "toyota", "foursquaregraph", "gowalla_7day", "gowalla_nevda"]:
            y_pred = self(batched_data)
            y_true = batched_data.y-1
            # y_true=(torch.cat([batched_data.y.unsqueeze(1).unsqueeze(1), batched_data.x[:,:-1]], dim=1))
            y_pred = self(batched_data)
        else:
            y_pred = self(batched_data)
            y_true = batched_data.y
            # y_true = batched_data.y[-len(y_pred):]
        return {
            "y_pred": y_pred,
            "y_true": y_true,
            "idx": batched_data.idx,
        }

    def test_epoch_end(self, outputs):
        queue_len = len(outputs)
        users_acc = {}
        for u in range(queue_len):  # Accumulate accuracy (NDCG, MRR) metrics for all users
            y_pred = outputs[u]["y_pred"][0]
            y_true = outputs[u]["y_true"]
            if u not in users_acc:
                users_acc[u] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            users_acc[u][0] += len(y_true)
            acc,ndcg = get_acc(y_true, y_pred)
            mrr=MRR_metric(y_true, y_pred)
            users_acc[u][1] += acc[2]  # accuracy for top-1 predicted elements
            users_acc[u][2] += acc[1]  # accuracy for top-5 predicted elements
            users_acc[u][3] += acc[0]  # accuracy for top-10 predicted elements
            users_acc[u][8] += acc[3]  # accuracy for top-20 predicted elements
            #ndcg
            users_acc[u][4] += ndcg[2]  # NDCG for top-1 predicted elements
            users_acc[u][5] += ndcg[1]  # NDCG for top-5 predicted elements
            users_acc[u][6] += ndcg[0]  # NDCG for top-10 predicted elements
            users_acc[u][9] += ndcg[3]  # NDCG for top-20 predicted elements
            users_acc[u][7] += mrr
        tmp_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tmp_mrr =[0.0]
        sum_test_samples = 0.0
        for u in users_acc:
            tmp_acc[0] = users_acc[u][1] + tmp_acc[0]  # top-1 ACC
            tmp_acc[1] = users_acc[u][2] + tmp_acc[1]  # top-5 ACC
            tmp_acc[2] = users_acc[u][3] + tmp_acc[2]  # top-10 ACC
            tmp_acc[3] = users_acc[u][4] + tmp_acc[3]  # top-1 NDCG
            tmp_acc[4] = users_acc[u][5] + tmp_acc[4]  # top-5 NDCG
            tmp_acc[5] = users_acc[u][6] + tmp_acc[5]  # top-10 NDCG
            sum_test_samples = sum_test_samples + users_acc[u][0]
            tmp_mrr[0] = users_acc[u][7] + tmp_mrr[0]
            tmp_acc[6] = users_acc[u][8] + tmp_acc[6]
            tmp_acc[7] = users_acc[u][9] + tmp_acc[7]
        avg_acc1 = (np.array(tmp_acc)/sum_test_samples)
        avg_acc = (np.array(tmp_acc)/sum_test_samples).tolist()
        avg_mrr = (np.array(tmp_mrr)/sum_test_samples).tolist()
        fin_acc=float(avg_acc1[0])
        try:
            self.log(
                "valid_" + self.metric,
                fin_acc,
                sync_dist=True,
            )
            acc_top1, acc_top5, acc_top10 = avg_acc[0][0], avg_acc[1][0], avg_acc[2][0]
            ndcg_top1, ndcg_top5, ndcg_top10 = avg_acc[3][0], avg_acc[4][0], avg_acc[5][0]
            print(f"ACC @1: {round(acc_top1, 4)}, @5: {round(acc_top5, 4)}, @10: {round(acc_top10, 4)}")
            print(f"NDCG @1: {round(ndcg_top1, 4)}, @5: {round(ndcg_top5, 4)}, @10: {round(ndcg_top10, 4)}")
            print(f"MRR: {round(avg_mrr[0], 4)}")
        except:
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay
        )
        lr_scheduler = {
            "scheduler": PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graphormer")
        parser.add_argument("--n_layers", type=int, default=12)
        parser.add_argument("--num_heads", type=int, default=32)
        parser.add_argument("--hidden_dim", type=int, default=512)
        parser.add_argument("--ffn_dim", type=int, default=512)
        parser.add_argument("--intput_dropout_rate", type=float, default=0.1)
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--attention_dropout_rate", type=float, default=0.1)
        parser.add_argument("--checkpoint_path", type=str, default="")
        parser.add_argument("--warmup_updates", type=int, default=60000)
        parser.add_argument("--tot_updates", type=int, default=1000000)
        parser.add_argument("--peak_lr", type=float, default=2e-4)
        parser.add_argument("--end_lr", type=float, default=1e-9)
        parser.add_argument("--edge_type", type=str, default="multi_hop")
        parser.add_argument("--validate", action="store_true", default=False)
        parser.add_argument("--test", action="store_true", default=False)
        parser.add_argument("--flag", action="store_true")
        parser.add_argument("--flag_m", type=int, default=3)
        parser.add_argument("--flag_step_size", type=float, default=1e-3)
        parser.add_argument("--flag_mag", type=float, default=1e-3)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask, 0)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm1 = nn.LayerNorm(hidden_size)
        self.ffn_norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None):
        # y = self.self_attention_norm(x)
        # y = self.self_attention(y, y, y, attn_bias, mask=mask)
        y = self.self_attention(x, x, x, attn_bias, mask=mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm1(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        x = self.ffn_norm2(x)
        return x

