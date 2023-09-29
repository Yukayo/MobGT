# coding: utf-8
from __future__ import print_function
from __future__ import division

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

from train_caller import run_simple, RnnParameterData, generate_input_history, markov, \
    generate_input_long_history, generate_input_long_history2
from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong



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

target_len=[]
def gen_poitimegraph(train_data):
    dict_data = {}
    # user_poigraph = copy.deepcopy(train_data)
    for user in train_data.keys():
        dict_data[user] = {}
        # index=len(data_train[user])
        for traj in train_data[user].keys():
            dict_data[user][traj] = {}
            # hist_traj = train_data[user][traj]['history_loc'].numpy()
            loc_len=len(train_data[user][traj]['target'])
            hist_traj = train_data[user][traj]['loc'][:-loc_len].numpy()
            # hist_traj = train_data[user][traj]['loc'].numpy()
            hist_traj = hist_traj.reshape(len(hist_traj)).tolist()
            current_traj = train_data[user][traj]['target'].numpy()
            current_traj = current_traj.reshape(len(current_traj)).tolist()
            target_len.append([len(current_traj),traj])
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
                    # latlon1=train_data[user][traj]['latlon'][i - 1]
                    # latlon2=train_data[user][traj]['latlon'][i]
                    # dist=LLs2Dist(latlon1[0],latlon1[1],latlon2[0],latlon2[1])
                    # if dist<=3 and dist>0:
                    #     dist_graph.loc[hist_traj[i-1], hist_traj[i]]=sigmoid(1/dist)
            dict_data[user][traj]['num_node']=len(set(hist_traj))
            dict_data[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(hist_traj))))
            # dist_graph=softmax(dist_graph)
            dict_data[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
            # dict_data[user][traj]['edge_type'] = copy.deepcopy(torch.FloatTensor((poi_graph+dist_graph).values))
            # dict_data[user][traj]['target'] = copy.deepcopy(torch.LongTensor(current_traj))
            dict_data[user][traj]['target'] = copy.deepcopy(torch.LongTensor([current_traj[-1]]))
            # dict_data[user][traj]['normalized']=0
            # dict_data[user][traj]['time'] = copy.deepcopy(time)


            # target未解决
    return dict_data

def gen_newpoitimegraph(train_data):
    dict_data = {}
    # user_poigraph = copy.deepcopy(train_data)
    for user in train_data.keys():
        dict_data[user] = {}
        # index=len(data_train[user])
        for traj in train_data[user].keys():
            dict_data[user][traj] = {}
            if traj==list(train_data[user].keys())[0]:
                traj_length=list(train_data[user].keys())[-1]
                # all_length=len(train_data[user][traj_length]['loc'].numpy())
                all_traj = train_data[user][traj_length]['loc'].numpy()
                all_traj = all_traj.reshape(len(all_traj)).tolist()
                num_poi = len(set(all_traj))
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(all_traj),
                                         columns=set(all_traj))
                dist_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                          index=set(all_traj),
                                          columns=set(all_traj))
                target_len=0
            # hist_traj = train_data[user][traj]['history_loc'].numpy()
            else:
                target_len=len(train_data[user][traj-1]['loc'].numpy())
            hist_traj = train_data[user][traj]['loc'].numpy()[-target_len:]
            hist_traj = hist_traj.reshape(len(hist_traj)).tolist()
            current_traj = train_data[user][traj]['target'].numpy()
            current_traj = current_traj.reshape(len(current_traj)).tolist()

            time=train_data[user][traj]['tim'][0:len(hist_traj)]

            for i in range(len(hist_traj)):
                if i>0:
                    poi_graph.loc[hist_traj[i-1], hist_traj[i]] += 1

            # for i in range(len(hist_traj)):
            #     if i>0:
            #         poi_graph.loc[hist_traj[i-1], hist_traj[i]] += 1
            #         latlon1=train_data[user][traj]['latlon'][i - 1]
            #         latlon2=train_data[user][traj]['latlon'][i]
            #         dist=LLs2Dist(latlon1[0],latlon1[1],latlon2[0],latlon2[1])
            #         if dist<=3 and dist>0:
            #             dist_graph.loc[hist_traj[i-1], hist_traj[i]]=sigmoid(1/dist)
            dict_data[user][traj]['num_node']=len(set(hist_traj))
            # dict_data[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(hist_traj))))
            dict_data[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list((all_traj))))
            # dist_graph=softmax(dist_graph)
            dict_data[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
            # dict_data[user][traj]['edge_type'] = copy.deepcopy(torch.FloatTensor((poi_graph+dist_graph).values))
            dict_data[user][traj]['target'] = copy.deepcopy(torch.LongTensor(current_traj))
            # dict_data[user][traj]['normalized']=0
            # dict_data[user][traj]['time'] = copy.deepcopy(time)


            # target未解决
    return dict_data


def gen_poigraph_d1124(train_data,mode):
    dict_data = {}
    train={}
    window=100
    # user_poigraph = copy.deepcopy(train_data)
    if mode=='train':
        for user in train_data.keys():
            dict_data[user]= train_data[user][len(train_data[user])]['loc'].numpy()
            dict_data[user]= dict_data[user].reshape(len(dict_data[user])).tolist()
            dict_data[user].append(train_data[user][len(train_data[user])]['target'][-1].tolist())
            train[user]={}
            traj_window=0
            for traj in range(len(dict_data[user])):
                if traj_window+window+1>=len(dict_data[user]):
                    break
                train[user][traj]={}
                current_traj=copy.deepcopy(dict_data[user][traj_window:traj_window+window])
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                for i in range(len(current_traj)):
                    if i>0:
                        poi_graph.loc[current_traj[i-1], current_traj[i]] += 1
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(torch.LongTensor([dict_data[user][traj_window+window+1]]))
                traj_window+=1
    else:
        for user in train_data.keys():
            train[user] = {}
            for traj in train_data[user]:
                target=train_data[user][traj]['target']
                if window-len(target)+1>=len(train_data[user][traj]['loc']):
                    current_traj = np.concatenate([train_data[user][traj]['loc'].reshape(
                        len(train_data[user][traj]['loc'])).numpy(), train_data[user][traj]['target'][:-1].numpy()],
                                                  axis=0).tolist()
                else:
                    current_traj=np.concatenate([train_data[user][traj]['loc'][-(window-len(target)+1):].reshape(window-len(target)+1).numpy(), train_data[user][traj]['target'][:-1].numpy()],axis=0).tolist()
                train[user][traj] = {}
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                for i in range(len(current_traj)):
                    if i > 0:
                        poi_graph.loc[current_traj[i - 1], current_traj[i]] += 1
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(
                    torch.LongTensor([train_data[user][traj]['target'][-1]]))
            # target未解决
    return train




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
        model.load_state_dict(torch.load("./exps/gowalla"+ "/res.m"))

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
    avg_acc_markov, users_acc_markov,avg5_acc_markov,avg10_acc_markov ,avg_ndcg_markov ,avg5_ndcg_markov ,avg10_ndcg_markov ,avg_mrr_markov  = markov(parameters, candidate)
    metrics['markov_acc'] = users_acc_markov

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

    # a=gen_poigraph_d1124(data_train,'train')
    # new_train_idx = {}
    # for user in a.keys():
    #     new_train_idx[user] = []
    #     for i in a[user].keys():
    #         new_train_idx[user].append(i)
    # b=gen_poigraph_d1124(data_test,'test')
    # new_test_idx = {}
    # for user in b.keys():
    #     new_test_idx[user]=[]
    #     for i in b[user].keys():
    #         new_test_idx[user].append(i)

    print(1)

    # with open('train.pickle', 'wb') as fw:
    #     pickle.dump(a, fw)

    # data_train = {x: data_train[x] for x in range(200)}
    # data_test = {x: data_test[x] for x in range(200)}
    # train_idx = {x: train_idx[x] for x in range(200)}
    # test_idx = {x: test_idx[x] for x in range(200)}

    print('users:{} markov:{} train:{} test:{}'.format(len(candidate), avg_acc_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    print('users:{} markov5acc:{} train:{} test:{}'.format(len(candidate), avg5_acc_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    print('users:{} markov10acc:{} train:{} test:{}'.format(len(candidate), avg10_acc_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    print('users:{} markovndcg:{} train:{} test:{}'.format(len(candidate), avg_ndcg_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    print('users:{} markov5ndcg:{} train:{} test:{}'.format(len(candidate), avg5_ndcg_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    print('users:{} markov10ndcg:{} train:{} test:{}'.format(len(candidate), avg10_ndcg_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    print('users:{} markovmrr:{} train:{} test:{}'.format(len(candidate), avg_mrr_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))                                                   
    SAVE_PATH = args.save_path
    tmp_path = 'checkpoint/'
    if os.path.exists(SAVE_PATH + tmp_path)==False:
        os.mkdir(SAVE_PATH + tmp_path)
    for epoch in range(parameters.epoch):
        st = time.time()
        if args.pretrain == 0:
            model, avg_loss = run_simple(data_train, train_idx, 'train', lr, parameters.clip, model, optimizer,
                                         criterion, parameters.model_mode)
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
            metrics['train_loss'].append(avg_loss)

        avg_loss, avg_acc, users_acc, avg_mrr = run_simple(data_test, test_idx, 'test', lr, parameters.clip, model,
                                                  optimizer, criterion, parameters.model_mode)
        print('==>Test [Acc@1, Acc@5, Acc@10, NCDG@1, NDCG@5, NDCG@10]:', avg_acc ,'Loss::{:.4f}'.format(avg_loss))
        print('==>Test [MRR]:', avg_mrr ,'Loss::{:.4f}'.format(avg_loss))
        avg_acc=np.mean(avg_acc[0])
        metrics['valid_loss'].append(avg_loss)
        metrics['accuracy'].append(avg_acc)
        metrics['valid_acc'][epoch] = users_acc

        save_name_tmp = 'ep_' + str(epoch) + '.m'
        torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)

        scheduler.step(avg_acc)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            load_epoch = np.argmax(metrics['accuracy'])
            load_name_tmp = 'ep_' + str(load_epoch) + '.m'
            model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
            print('load epoch={} model state'.format(load_epoch))
        if epoch == 0:
            print('single epoch time cost:{}'.format(time.time() - st))
        if lr <= 0.9 * 1e-5:
            break #学习率小于某个值就停止训练
        if args.pretrain == 1:
            break

    mid = np.argmax(metrics['accuracy'])
    avg_acc = metrics['accuracy'][mid]
    load_name_tmp = 'ep_' + str(mid) + '.m'
    model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
    save_name = 'res'
    json.dump({'args': argv, 'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)
    metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy': []}
    for key in metrics_view:
        metrics_view[key] = metrics[key]
    json.dump({'args': argv, 'metrics': metrics_view}, fp=open(SAVE_PATH + save_name + '.txt', 'w'), indent=4)
    torch.save(model.state_dict(), SAVE_PATH + save_name + '.m')

    for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
        for name in files:
            remove_path = os.path.join(rt, name)
            os.remove(remove_path)
    os.rmdir(SAVE_PATH + tmp_path)

    return avg_acc


def load_pretrained_model(config):
    res = json.load(open("./exps/gowalla" + "/res.txt"))
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=500, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
    parser.add_argument('--voc_emb_size', type=int, default=50, help="words embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    # parser.add_argument('--data_name', type=str, default='toyota_OD_vis')
    # parser.add_argument('--data_name', type=str, default='foursquare _deepmove')
    # parser.add_argument('--data_name', type=str, default='toyota_seq')
    parser.add_argument('--data_name', type=str, default='gowalla_nvda')
    parser.add_argument('--learning_rate', type=float, default=5 * 1e-4)
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=20)
    parser.add_argument('--history_mode', type=str, default='avg', choices=['max', 'avg', 'whole'])
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--attn_type', type=str, default='dot', choices=['general', 'concat', 'dot'])   #注意力机制的打分方法
    parser.add_argument('--data_path', type=str, default='../../dataset/baseline_models_dataset/LSTM/')
    parser.add_argument('--save_path', type=str, default='./exps/gowalla/')
    parser.add_argument('--model_mode', type=str, default='simple',
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()
    if args.pretrain == 1:
        args = load_pretrained_model(args)

    ours_acc = run(args)
