from __future__ import print_function
from __future__ import division

# from torch_geometric.datasets import KarateClub
import networkx as nx
import matplotlib.pyplot as plt



# coding: utf-8

import torch
import torch.nn as nn
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

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

from train import run_simple, RnnParameterData, generate_input_history, markov, \
    generate_input_long_history, generate_input_long_history2
# from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong
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
    np.random.seed(43)
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
                if split == 'train_d10':
                    split = 'train'
                if split == 'test_d10':
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
                pp += 1
                mol = mols[idx[0]][idx[1]]
                x = mol['node_name'].to(torch.long).view(-1, 1)
                y = mol['target'].to(torch.long)

                adj = mol['edge_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.float)

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
    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.float)
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


def LLs2Dist(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(lat1 * math.pi / 180.0) * math.cos(lat2 * math.pi / 180.0) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist = R * c
    return dist

def softmax(scores):
    score=copy.deepcopy(scores)
    score -= np.max(scores)
    p = (np.exp(score).T / np.sum(np.exp(score),axis=1)).T
    return p

def sigmoid(x):
	return 1/(1+np.exp(-x))

def gen_poitimegraph(train_data):
    dict_data = {}
    # user_poigraph = copy.deepcopy(train_data)
    for user in train_data.keys():
        dict_data[user] = {}
        # index=len(data_train[user])
        for traj in train_data[user].keys():
            dict_data[user][traj] = {}
            hist_traj = train_data[user][traj]['history_loc'].numpy()
            hist_traj = hist_traj.reshape(len(hist_traj)).tolist()
            current_traj = train_data[user][traj]['target'].numpy()
            current_traj = current_traj.reshape(len(current_traj)).tolist()
            num_poi = len(set(hist_traj))
            time=train_data[user][traj]['tim'][0:len(hist_traj)]
            poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                        index=set(hist_traj),
                                        columns=set(hist_traj))
            dist_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                     index=set(hist_traj),
                                     columns=set(hist_traj))
            for i in range(len(hist_traj)):
                if i>0:
                    poi_graph.loc[hist_traj[i-1], hist_traj[i]] += 1
                    latlon1=train_data[user][traj]['latlon'][i - 1]
                    latlon2=train_data[user][traj]['latlon'][i]
                    dist=LLs2Dist(latlon1[0],latlon1[1],latlon2[0],latlon2[1])
                    if dist<=3 and dist>0:
                        dist_graph.loc[hist_traj[i-1], hist_traj[i]]=sigmoid(1/dist)
            dict_data[user][traj]['num_node']=len(set(hist_traj))
            dict_data[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(hist_traj))))
            # dist_graph=softmax(dist_graph)
            dict_data[user][traj]['edge_type'] = copy.deepcopy(torch.FloatTensor((poi_graph+dist_graph).values))
            dict_data[user][traj]['target'] = copy.deepcopy(torch.LongTensor(current_traj))
            # dict_data[user][traj]['normalized']=0
            # dict_data[user][traj]['time'] = copy.deepcopy(time)


            # target未解决
    return dict_data

def plot_graph(edge, x, y):
    # edge, x, y 每个维度都为2，其中第一维度是name，第二个维度是data
    # x表示的是结点，y表示的标签，edge表示的连边, 由两个维度的tensor构成
    x_np = x[1].numpy()
    y_np = y[1].numpy()
    g = nx.DiGraph()
    src = edge[0].numpy()
    dst = edge[1].numpy()
    edgelist = zip(src, dst)
    for i, j in edgelist:
        g.add_edge(i, j)
    nx.draw(g, with_labels=g.nodes)
    # plt.savefig('test.png')
    plt.show()

def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path)
    argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}
    print('*' * 15 + 'start training' + '*' * 15)
    print('model_mode:{} history_mode:{} users:{}'.format(
        parameters.model_mode, parameters.history_mode, parameters.uid_size))

    if parameters.model_mode in ['simple', 'simple_long']:
        model = TrajPreSimple(parameters=parameters).cuda()
    elif parameters.model_mode == 'attn_avg_long_user':
        model = TrajPreAttnAvgLongUser(parameters=parameters).cuda()
    elif parameters.model_mode == 'attn_local_long':
        model = TrajPreLocalAttnLong(parameters=parameters).cuda()
    if args.pretrain == 1:
        # model.load_state_dict(torch.load("./DeepMove-master/DeepMove-master/pretrain/" + args.model_mode + "/res.m"))
        model.load_state_dict(torch.load("../pretrain/" + args.model_mode + "/res.m"))

    if 'max' in parameters.model_mode:
        parameters.history_mode = 'max'
    elif 'avg' in parameters.model_mode:
        parameters.history_mode = 'avg'
    else:
        parameters.history_mode = 'whole'

    criterion = nn.NLLLoss().cuda() #为什么不是CrossEntropyLossCrossEntropyLoss,是因为模型最后输出的评分是已经经过log(softmax)过的，所以这个直接测试(scores,target)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
                           weight_decay=parameters.L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
                                                     factor=parameters.lr_decay, threshold=1e-3) #调整学习率
    # mode(str) -min , max 之一。 min模式下，当监控的数量停止减少时，lr会减少；在max 模式下，当监控的数量停止增加时，它将减少。默认值：‘min’。
    # factor(float) -将降低学习率的因子。 new_lr = lr * 因子。默认值：0.1。
    # patience(int) -没有改善的时期数，之后学习率将降低。例如，如果 patience = 2 ，那么我们将忽略前 2 个没有改善的 epoch，并且仅在第 3 个 epoch 之后损失仍然没有改善的情况下降低 LR。默认值：10。
    # threshold(float) -衡量新的最佳阈值，只关注重大变化。默认值：1e-4。
    # threshold_mode(str) -rel , abs 之一。在rel 模式中，dynamic_threshold = ‘max’ 模式中的最佳 * (1 + 阈值) 或 min 模式中的最佳 * (1 - 阈值)。在abs 模式下，dynamic_threshold = max 模式下的最佳 + 阈值或 min 模式下的最佳 - 阈值。默认值：‘rel’。
    # cooldown(int) -在 lr 减少后恢复正常操作之前要等待的 epoch 数。默认值：0。
    # min_lr(float或者list) -标量或标量列表。所有参数组或每个组的学习率的下限。默认值：0。
    # eps(float) -应用于 lr 的最小衰减。如果新旧 lr 之间的差异小于 eps，则忽略更新。默认值：1e-8。
    # verbose(bool) -如果 True ，每次更新都会向标准输出打印一条消息。默认值：False。
    # 当指标停止改进时降低学习率。一旦学习停滞，模型通常会受益于将学习率降低 2-10 倍。该调度程序读取一个指标数量，如果在 ‘patience’ 的 epoch 数量上没有看到任何改进，则学习率会降低。

    lr = parameters.lr
    metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {}}

    candidate = parameters.data_neural.keys()
    avg_acc_markov, users_acc_markov,avg5_acc_markov = markov(parameters, candidate)
    metrics['markov_acc'] = users_acc_markov

    dataset = {
        # "num_class": 1,
        # loss和deepmove暂且一致
        "loss_fn": nn.NLLLoss().cuda(),
        # "metric": "cross_entropy",
        "metric": "NLLLoss",
        "metric_mode": "min",
        "train_dataset": MyFoursquareDataset(
            subset=False, root="../dataset/foursquare", split="train"
        ),
        "test_dataset": MyFoursquareDataset(
            subset=False, root="../dataset/foursquare", split="test"
        ),

        # "max_node": 2708,
    }

    if 'long' in parameters.model_mode:
        long_history = True
    else:
        long_history = False

    if long_history is False:
        data_train, train_idx = generate_input_history(parameters.data_neural, 'train', mode2=parameters.history_mode,
                                                       candidate=candidate)
        data_test, test_idx = generate_input_history(parameters.data_neural, 'test', mode2=parameters.history_mode,
                                                     candidate=candidate)
    elif long_history is True:
        if parameters.model_mode == 'simple_long':
            data_train, train_idx = generate_input_long_history2(parameters.data_neural, 'train', candidate=candidate)
            data_test, test_idx = generate_input_long_history2(parameters.data_neural, 'test', candidate=candidate)
        else:
            data_train, train_idx = generate_input_long_history(parameters.data_neural, 'train', candidate=candidate)
            data_test, test_idx = generate_input_long_history(parameters.data_neural, 'test', candidate=candidate)

    a=gen_poitimegraph(data_train)
    b=gen_poitimegraph(data_test)
    # print(1)

    # with open('train.pickle', 'wb') as fw:
    #     pickle.dump(a, fw)

    # data_train = {x: data_train[x] for x in range(200)}
    # data_test = {x: data_test[x] for x in range(200)}
    # train_idx = {x: train_idx[x] for x in range(200)}
    # test_idx = {x: test_idx[x] for x in range(200)}

    edge_index, x, y = data_train['edge_index'], data_train['x'], data_train['y']
    plot_graph(edge_index, x, y)

    print('users:{} markov:{} train:{} test:{}'.format(len(candidate), avg_acc_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))


def load_pretrained_model(config):
    res = json.load(open("../pretrain/" + config.model_mode + "/res.txt"))
    args = Settings(config, res["args"])
    return args


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["data_name"]
        self.epoch_max = res["epoch_max"]
        self.learning_rate = res["learning_rate"]
        self.lr_step = res["lr_step"]
        self.lr_decay = res["lr_decay"]
        self.clip = res["clip"]
        self.dropout_p = res["dropout_p"]
        self.rnn_type = res["rnn_type"]
        self.attn_type = res["attn_type"]
        self.L2 = res["L2"]
        self.history_mode = res["history_mode"]
        self.model_mode = res["model_mode"]
        self.optim = res["optim"]
        self.hidden_size = res["hidden_size"]
        self.tim_emb_size = res["tim_emb_size"]
        self.loc_emb_size = res["loc_emb_size"]
        self.uid_emb_size = res["uid_emb_size"]
        self.voc_emb_size = res["voc_emb_size"]
        self.pretrain = 1


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=500, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
    parser.add_argument('--voc_emb_size', type=int, default=50, help="words embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='toyota_OD_vis')
    # parser.add_argument('--data_name', type=str, default='foursquare_cut_one_day')
    # parser.add_argument('--data_name', type=str, default='foursquarelatlon')
    parser.add_argument('--learning_rate', type=float, default=5 * 1e-4)
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=20)
    parser.add_argument('--history_mode', type=str, default='avg', choices=['max', 'avg', 'whole'])
    parser.add_argument('--rnn_type', type=str, default='GRU', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--attn_type', type=str, default='dot', choices=['general', 'concat', 'dot'])   #注意力机制的打分方法
    parser.add_argument('--data_path', type=str, default='../data/')
    # parser.add_argument('--data_path', type=str, default='/home/xuxh/xuxh/LSTPM-master/LSTPM-master/train/')
    parser.add_argument('--save_path', type=str, default='../results/')
    parser.add_argument('--model_mode', type=str, default='attn_local_long',
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()
    if args.pretrain == 1:
        args = load_pretrained_model(args)

    ours_acc = run(args)


