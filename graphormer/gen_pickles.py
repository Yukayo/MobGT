# coding: utf-8
from __future__ import print_function
from __future__ import division

from train import RnnParameterData, generate_input_history
max_num=0
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

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

def generate_input_long_history_graph(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    # data_neural = {x: data_neural[x] for x in range(200)}
    # candidate = data_neural.keys()
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            trace['user'] = Variable(torch.LongTensor([u]))
            # if mode == 'train' and c == 0:
            #     continue
            session = sessions[i] #取每个session(第一个session没取，被上面的if判断跳过了)
            target = np.array([session[-1][0]]) #一个session有11个POI，这就相当于取了前10个
            # target = np.array([s[2] for s in session[1:]])

            history = []
            latlon=[]
            pname=[]
            # if mode == 'test':
            #     test_id = data_neural[u]['train']
            #     for tt in test_id:
            #         history.extend([(s[0], s[1]) for s in sessions[tt]])
                    # history.extend([(s[2], s[1]) for s in sessions[tt]])
                    # latlon.extend([(s[3], s[2]) for s in sessions[tt]])
                    # pname.extend([(s[4]) for s in sessions[tt]])
            # for j in range(c):
            history.extend([(s[0], s[1], s[-1]) for s in sessions[train_id[c]][:-1]   ])
                # history.extend([(s[2], s[1]) for s in sessions[train_id[j]]])
                # latlon.extend([(s[3], s[2]) for s in sessions[train_id[j]]])
                # pname.extend([(s[4]) for s in sessions[train_id[j]]])

            history_tim = [t[1] for t in history] #取时间
            history_count = [1]
            last_t = history_tim[0]
            count = 1

            cat = [c[-1] for c in history]
            cat = np.reshape(np.array([s[-1] for s in history]), (len(history), 1))
            trace['cat'] = Variable(torch.LongTensor(cat))
            #看起来是像在统计相同的时间段出现了多少次
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1
            #看起来是在把loc和time分成一个一个的,like:[[1],[1],[1],[1],[2],[2],[1],[3],[4],[1],[5]]
            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            #x = Variable(tensor, requires_grad = True)
            #Varibale包含三个属性：
            # data：存储了Tensor，是本体的数据
            # grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
            # grad_fn：指向Function对象，用于反向传播的梯度计算之用
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))  #torch.LongTensor是64位整型
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_count'] = history_count

            loc_tim = history
            # loc_tim.extend([(s[2], s[1]) for s in session[:-1]])
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]]) #看起来是把每次的session加上形成一个大序列

            # latlon.extend([(s[3], s[2]) for s in session[:-1]])
            # pname.extend([(s[4]) for s in session[:-1]])

            #又是一个逐一分割操作
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            latlon_np = np.reshape(np.array([s for s in latlon]), (len(latlon), 2))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target))
            # trace['latlon'] = Variable(torch.FloatTensor(latlon_np))
            # trace['pname'] = Variable(pname)
            data_train[u][i] = trace
            # data_train[u][i]['pname']=pname
        train_idx[u] = train_id
    return data_train, train_idx


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
    dict_time = {}
    train={}
    window=10
    padding=0
    len_num=10 #window 10 len_num:10
    global max_num
    # user_poigraph = copy.deepcopy(train_data)
    if mode=='train':
        for user in train_data.keys():
            dict_data[user]= train_data[user][len(train_data[user])]['loc'].numpy()
            dict_data[user]= dict_data[user].reshape(len(dict_data[user])).tolist()
            dict_data[user].append(train_data[user][len(train_data[user])]['target'][-1].tolist())

            dict_time[user] = train_data[user][len(train_data[user])]['tim'].numpy()
            dict_time[user] = dict_time[user].reshape(len(dict_time[user])).tolist()

            train[user]={}
            traj_window=0
            for traj in range(len(dict_data[user])):
                if traj_window+window+1>=len(dict_data[user]):
                    break
                train[user][traj]={}
                current_traj=copy.deepcopy(dict_data[user][traj_window:traj_window+window])
                current_time = copy.deepcopy(dict_time[user][traj_window:traj_window + window])
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                temp_time={}
                tensor_list = []
                for i in range(len(current_traj)):
                    if i>0:
                        poi_graph.loc[current_traj[i-1], current_traj[i]] += 1
                    if bool(temp_time.get(current_traj[i]))==False:
                        temp_time[current_traj[i]]=[]
                        temp_time[current_traj[i]].append(current_time[i])
                    else:
                        temp_time[current_traj[i]].append(current_time[i])
                max_len=len(temp_time[max(temp_time, key=lambda k: len(temp_time[k]))])
                if max_num<max_len:
                    max_num=copy.deepcopy(max_len)
                for i in temp_time.keys():
                    if len(temp_time[i])<len_num:
                        for j in range(len_num-len(temp_time[i])):
                            temp_time[i].append(padding)
                    else:
                        continue
                for i in temp_time.keys():
                    tensor_list.append(temp_time[i])
                    # tensor_list.append(torch.LongTensor(temp_time[i]))
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(torch.LongTensor([dict_data[user][traj_window+window+1]]))
                train[user][traj]['time'] = copy.deepcopy(torch.LongTensor(tensor_list))
                # train[user][traj]['time'] = copy.deepcopy(torch.stack(tensor_list,0))
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
                    current_time=train_data[user][traj]['tim'].reshape(
                        len(train_data[user][traj]['tim'])).numpy().tolist()
                else:
                    if window>len(train_data[user][traj]['tim']):
                        current_time = train_data[user][traj]['tim'].reshape(len(train_data[user][traj]['tim'])).numpy().tolist()
                    else:
                        current_time=train_data[user][traj]['tim'][-window:].reshape(window).numpy().tolist()
                    current_traj = np.concatenate([train_data[user][traj]['loc'][-(window - len(target) + 1):].reshape(
                        window - len(target) + 1).numpy(), train_data[user][traj]['target'][:-1].numpy()],
                                                  axis=0).tolist()
                train[user][traj] = {}
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                temp_time = {}
                tensor_list = []
                for i in range(len(current_traj)):
                    if i > 0:
                        poi_graph.loc[current_traj[i - 1], current_traj[i]] += 1
                    if bool(temp_time.get(current_traj[i])) == False:
                        temp_time[current_traj[i]] = []
                        temp_time[current_traj[i]].append(current_time[i])
                    else:
                        if i<len(current_time):
                            temp_time[current_traj[i]].append(current_time[i])
                max_len = len(temp_time[max(temp_time, key=lambda k: len(temp_time[k]))])
                if max_num<max_len:
                    max_num=copy.deepcopy(max_len)
                for i in temp_time.keys():
                    if len(temp_time[i]) < len_num:
                        for j in range(len_num - len(temp_time[i])):
                            temp_time[i].append(padding)
                    else:
                        continue
                for i in temp_time.keys():
                    tensor_list.append(temp_time[i])
                    # tensor_list.append(torch.LongTensor(temp_time[i]))
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(
                    torch.LongTensor([train_data[user][traj]['target'][-1]]))
                train[user][traj]['time'] = copy.deepcopy(torch.LongTensor(tensor_list))
                # train[user][traj]['time'] = copy.deepcopy(torch.stack(tensor_list, 0))
    return train

def gen_poigraph_d1124_max(train_data,mode):
    dict_data = {}
    dict_time = {}
    train={}
    window=10
    padding=0
    len_num=10 #window 10 len_num:10
    global max_num
    # user_poigraph = copy.deepcopy(train_data)
    if mode=='train':
        for user in train_data.keys():
            dict_data[user]= train_data[user][len(train_data[user])]['loc'].numpy()
            dict_data[user]= dict_data[user].reshape(len(dict_data[user])).tolist()
            dict_data[user].append(train_data[user][len(train_data[user])]['target'][-1].tolist())

            dict_time[user] = train_data[user][len(train_data[user])]['tim'].numpy()
            dict_time[user] = dict_time[user].reshape(len(dict_time[user])).tolist()

            train[user]={}
            traj_window=0
            for traj in range(len(dict_data[user])):
                if traj_window+window+1>=len(dict_data[user]):
                    break
                train[user][traj]={}
                current_traj=copy.deepcopy(dict_data[user][traj_window:traj_window+window])
                current_time = copy.deepcopy(dict_time[user][traj_window:traj_window + window])
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                temp_time={}
                tensor_list = []
                for i in range(len(current_traj)):
                    if i>0:
                        poi_graph.loc[current_traj[i-1], current_traj[i]] += 1
                    temp_time[current_traj[i]] = copy.deepcopy(current_time[i])
                for i in temp_time.keys():
                    tensor_list.append(temp_time[i])
                    # tensor_list.append(torch.LongTensor(temp_time[i]))
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(torch.LongTensor([dict_data[user][traj_window+window+1]]))
                train[user][traj]['time'] = copy.deepcopy(torch.LongTensor(tensor_list))
                # train[user][traj]['time'] = copy.deepcopy(torch.stack(tensor_list,0))
                traj_window+=1
    else:
        for user in train_data.keys():
            train[user] = {}
            for traj in train_data[user]:
                target=train_data[user][traj]['target']
                if window>=len(train_data[user][traj]['loc']):
                    current_traj = train_data[user][traj]['loc'].tolist()
                    current_time=train_data[user][traj]['tim'].reshape(
                        len(train_data[user][traj]['tim'])).numpy().tolist()
                else:
                    current_time=train_data[user][traj]['tim'][-window:].reshape(window).numpy().tolist()
                    current_traj = train_data[user][traj]['loc'][-window:].reshape(window).numpy().tolist()
                train[user][traj] = {}
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                temp_time = {}
                tensor_list = []
                for i in range(len(current_traj)):
                    if i > 0:
                        poi_graph.loc[current_traj[i - 1], current_traj[i]] += 1
                    # if i<len(current_time):
                    temp_time[current_traj[i]]=copy.deepcopy(current_time[i])
                for i in temp_time.keys():
                    tensor_list.append(temp_time[i])
                    # tensor_list.append(torch.LongTensor(temp_time[i]))
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(
                    torch.LongTensor([train_data[user][traj]['target'][-1]]))
                train[user][traj]['time'] = copy.deepcopy(torch.LongTensor(tensor_list))
                # train[user][traj]['time'] = copy.deepcopy(torch.stack(tensor_list, 0))
    return train


def gen_poigraph_d1124_max_reindex(train_data,mode):
    dict_data = {}
    dict_time = {}
    train={}
    window=10
    padding=0
    len_num=10 #window 10 len_num:10
    global max_num
    # user_poigraph = copy.deepcopy(train_data)
    if mode=='train':
        for user in train_data.keys():
            dict_data[user]= train_data[user][len(train_data[user])]['loc'].numpy()
            dict_data[user]= dict_data[user].reshape(len(dict_data[user])).tolist()
            dict_data[user].append(train_data[user][len(train_data[user])]['target'][-1].tolist())

            dict_time[user] = train_data[user][len(train_data[user])]['tim'].numpy()
            dict_time[user] = dict_time[user].reshape(len(dict_time[user])).tolist()

            train[user]={}
            traj_window=0
            for traj in range(len(dict_data[user])):
                if traj_window+window+1>=len(dict_data[user]):
                    break
                train[user][traj]={}
                current_traj=copy.deepcopy(dict_data[user][traj_window:traj_window+window])
                current_time = copy.deepcopy(dict_time[user][traj_window:traj_window + window])
                df=pd.DataFrame(current_traj)
                df=df.drop_duplicates(keep='last')
                newtraj_list=df.to_numpy().reshape(len(df.to_numpy())).tolist()
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                temp_time={}
                tensor_list = []
                for i in range(len(current_traj)):
                    if i>0:
                        poi_graph.loc[current_traj[i-1], current_traj[i]] += 1
                    temp_time[current_traj[i]] = copy.deepcopy(current_time[i])
                for i in newtraj_list:
                    tensor_list.append(temp_time[i])
                    # tensor_list.append(torch.LongTensor(temp_time[i]))
                poi_graph = poi_graph[newtraj_list]
                poi_graph=poi_graph.reindex(newtraj_list)
                # train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(newtraj_list))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(torch.LongTensor([dict_data[user][traj_window+window+1]]))
                train[user][traj]['time'] = copy.deepcopy(torch.LongTensor(tensor_list))
                # train[user][traj]['time'] = copy.deepcopy(torch.stack(tensor_list,0))
                traj_window+=1
    else:
        for user in train_data.keys():
            train[user] = {}
            for traj in train_data[user]:
                target=train_data[user][traj]['target']
                if window>=len(train_data[user][traj]['loc']):
                    current_traj = train_data[user][traj]['loc'].reshape(
                        len(train_data[user][traj]['loc'])).numpy().tolist()
                    current_time=train_data[user][traj]['tim'].reshape(
                        len(train_data[user][traj]['tim'])).numpy().tolist()
                else:
                    current_time=train_data[user][traj]['tim'][-window:].reshape(window).numpy().tolist()
                    current_traj = train_data[user][traj]['loc'][-window:].reshape(window).numpy().tolist()
                train[user][traj] = {}
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                df = pd.DataFrame(current_traj)
                df = df.drop_duplicates(keep='last')
                newtraj_list = df.to_numpy().reshape(len(df.to_numpy())).tolist()
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                temp_time = {}
                tensor_list = []
                for i in range(len(current_traj)):
                    if i > 0:
                        poi_graph.loc[current_traj[i - 1], current_traj[i]] += 1
                    # if i<len(current_time):
                    temp_time[current_traj[i]]=copy.deepcopy(current_time[i])
                for i in newtraj_list:
                    tensor_list.append(temp_time[i])
                    # tensor_list.append(torch.LongTensor(temp_time[i]))
                poi_graph = poi_graph[newtraj_list]
                poi_graph = poi_graph.reindex(newtraj_list)
                # train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(newtraj_list))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(
                    torch.LongTensor([train_data[user][traj]['target'][-1]]))
                train[user][traj]['time'] = copy.deepcopy(torch.LongTensor(tensor_list))
                # train[user][traj]['time'] = copy.deepcopy(torch.stack(tensor_list, 0))
    return train

def gen_poigraph_d1124_avg_reindex(train_data,mode):
    dict_data = {}
    dict_time = {}
    train={}
    window=50
    padding=0
    len_num=50 #window 10 len_num:10
    global max_num
    # user_poigraph = copy.deepcopy(train_data)
    if mode=='train':
        for user in train_data.keys():
            dict_data[user]= train_data[user][len(train_data[user])]['loc'].numpy()
            dict_data[user]= dict_data[user].reshape(len(dict_data[user])).tolist()
            dict_data[user].append(train_data[user][len(train_data[user])]['target'][-1].tolist())

            dict_time[user] = train_data[user][len(train_data[user])]['tim'].numpy()
            dict_time[user] = dict_time[user].reshape(len(dict_time[user])).tolist()

            train[user]={}
            traj_window=0
            for traj in range(len(dict_data[user])):
                if traj_window+window+1>=len(dict_data[user]):
                    break
                train[user][traj]={}
                current_traj=copy.deepcopy(dict_data[user][traj_window:traj_window+window])
                current_time = copy.deepcopy(dict_time[user][traj_window:traj_window + window])
                df=pd.DataFrame(current_traj)
                df=df.drop_duplicates(keep='last')
                newtraj_list=df.to_numpy().reshape(len(df.to_numpy())).tolist()
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                temp_time={}
                tensor_list = {}
                for i in range(len(current_traj)):
                    if i>0:
                        poi_graph.loc[current_traj[i-1], current_traj[i]] += 1
                    if bool(temp_time.get(current_traj[i]))==False:
                        temp_time[current_traj[i]]=[]
                        temp_time[current_traj[i]].append(current_time[i])
                    else:
                        temp_time[current_traj[i]].append(current_time[i])
                # max_len=len(temp_time[max(temp_time, key=lambda k: len(temp_time[k]))])
                # if max_num<max_len:
                #     max_num=copy.deepcopy(max_len)
                for i in temp_time.keys():
                    if len(temp_time[i])<len_num:
                        for j in range(len_num-len(temp_time[i])):
                            temp_time[i].append(padding)
                    else:
                        continue
                for i in temp_time.keys():
                    tensor_list[i]=copy.deepcopy(temp_time[i])
                newtensor_list=[]
                for i in range(len(tensor_list)):
                    newtensor_list.append(tensor_list[newtraj_list[i]])
                poi_graph = poi_graph[newtraj_list]
                poi_graph=poi_graph.reindex(newtraj_list)
                # train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(newtraj_list))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(torch.LongTensor([dict_data[user][traj_window+window+1]]))
                train[user][traj]['time'] = copy.deepcopy(torch.LongTensor(newtensor_list))
                # train[user][traj]['time'] = copy.deepcopy(torch.stack(tensor_list,0))
                traj_window+=1
    else:
        for user in train_data.keys():
            train[user] = {}
            for traj in train_data[user]:
                target=train_data[user][traj]['target']
                if window>=len(train_data[user][traj]['loc']):
                    current_traj = train_data[user][traj]['loc'].reshape(
                        len(train_data[user][traj]['loc'])).numpy().tolist()
                    current_time=train_data[user][traj]['tim'].reshape(
                        len(train_data[user][traj]['tim'])).numpy().tolist()
                else:
                    current_time=train_data[user][traj]['tim'][-window:].reshape(window).numpy().tolist()
                    current_traj = train_data[user][traj]['loc'][-window:].reshape(window).numpy().tolist()
                train[user][traj] = {}
                train[user][traj]['num_node'] = len(set(current_traj))
                num_poi = len(set(current_traj))
                df = pd.DataFrame(current_traj)
                df = df.drop_duplicates(keep='last')
                newtraj_list = df.to_numpy().reshape(len(df.to_numpy())).tolist()
                poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                         index=set(current_traj),
                                         columns=set(current_traj))
                temp_time = {}
                tensor_list = {}
                for i in range(len(current_traj)):
                    if i > 0:
                        poi_graph.loc[current_traj[i - 1], current_traj[i]] += 1
                    if bool(temp_time.get(current_traj[i])) == False:
                        temp_time[current_traj[i]] = []
                        temp_time[current_traj[i]].append(current_time[i])
                    else:
                        temp_time[current_traj[i]].append(copy.deepcopy(current_time[i]))
                for i in temp_time.keys():
                    if len(temp_time[i]) < len_num:
                        for j in range(len_num - len(temp_time[i])):
                            temp_time[i].append(padding)
                    else:
                        continue
                for i in temp_time.keys():
                    tensor_list[i]=copy.deepcopy(temp_time[i])
                newtensor_list = []
                for i in range(len(tensor_list)):
                    newtensor_list.append(tensor_list[newtraj_list[i]])
                poi_graph = poi_graph[newtraj_list]
                poi_graph = poi_graph.reindex(newtraj_list)
                # train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(list(set(current_traj))))
                train[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(newtraj_list))
                train[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
                train[user][traj]['target'] = copy.deepcopy(
                    torch.LongTensor([train_data[user][traj]['target'][-1]]))
                train[user][traj]['time'] = copy.deepcopy(torch.LongTensor(newtensor_list))
                # train[user][traj]['time'] = copy.deepcopy(torch.stack(tensor_list, 0))
    return train

def gen_poigraph_d1212_nyc_avg_maxtime(train_data,mode):
    dict_data = {}
    dict_time = {}
    train={}
    dict_data = {}
    # user_poigraph = copy.deepcopy(train_data)
    for user in train_data.keys():
        dict_data[user] = {}
        dict_time[user] = {}
        # index=len(data_train[user])
        for traj in train_data[user].keys():
            # if len(train_data[user][traj]['history_loc'])!=10:
            #     break
            dict_data[user][traj] = {}
            # hist_traj = train_data[user][traj]['history_loc'].numpy()
            loc_len = len(train_data[user][traj]['target'])
            hist_traj = train_data[user][traj]['history_loc'].numpy()
            hist_traj = hist_traj.reshape(len(hist_traj)).tolist()
            current_traj = train_data[user][traj]['target'].numpy()
            # current_traj = current_traj.reshape(len(current_traj)).tolist()
            # target_len.append([len(current_traj), traj])
            num_poi = len(set(hist_traj))

            poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                     index=set(hist_traj),
                                     columns=set(hist_traj))
            dist_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                      index=set(hist_traj),
                                      columns=set(hist_traj))
            current_time = copy.deepcopy(train_data[user][traj]['tim'][:len(hist_traj)].numpy())
            temp_time = {}
            tensor_list = {}
            for i in range(len(hist_traj)):
                if i > 0:
                    poi_graph.loc[hist_traj[i - 1], hist_traj[i]] += 1
                    # latlon1=train_data[user][traj]['latlon'][i - 1]
                    # latlon2=train_data[user][traj]['latlon'][i]
                    # dist=LLs2Dist(latlon1[0],latlon1[1],latlon2[0],latlon2[1])
                    # if dist<=3 and dist>0:
                    #     dist_graph.loc[hist_traj[i-1], hist_traj[i]]=sigmoid(1/dist)
                temp_time[hist_traj[i]] = copy.deepcopy(current_time[i])
            df = pd.DataFrame(hist_traj)
            df = df.drop_duplicates(keep='last')
            newtraj_list = df.to_numpy().reshape(len(df.to_numpy())).tolist()
            for i in temp_time.keys():
                tensor_list[i] = copy.deepcopy(temp_time[i])
            newtensor_list = []
            for i in range(len(tensor_list)):
                newtensor_list.append(tensor_list[newtraj_list[i]])
            poi_graph = poi_graph[newtraj_list]
            poi_graph = poi_graph.reindex(newtraj_list)



            dict_data[user][traj]['num_node'] = len(set(hist_traj))
            dict_data[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(newtraj_list))
            dict_data[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
            # dict_data[user][traj]['edge_type'] = copy.deepcopy(torch.FloatTensor((poi_graph+dist_graph).values))
            dict_data[user][traj]['target'] = copy.deepcopy(torch.LongTensor([current_traj[-1]]))
            # dict_data[user][traj]['target'] = copy.deepcopy(torch.LongTensor(newtraj_list[1:] + [current_traj[-1]]))
            dict_data[user][traj]['time']  = copy.deepcopy(torch.LongTensor(newtensor_list))
    return dict_data




def gen_poigraph_d1228_nyc_avg_maxtime(train_data, mode):
    dict_data = {}
    dict_time = {}
    train={}
    dict_data = {}
    # user_poigraph = copy.deepcopy(train_data)
    for user in train_data.keys():
        dict_data[user] = {}
        dict_time[user] = {}
        # index=len(data_train[user])
        for traj in train_data[user]['sessions'].keys():
            if mode=='train':
                if traj not in train_data[user]['train']:
                    break
            else:
                if traj not in train_data[user]['test']:
                    continue
                if traj in train_data[user]['train']:
                    continue

            dict_data[user][traj] = {}
            loc_len = len(train_data[user]['sessions'][traj])
            all_traj = [poi[0] for poi in train_data[user]['sessions'][traj]]
            lat_lon=[[poi[3],poi[2]] for poi in train_data[user]['sessions'][traj]]

            hist_traj = all_traj[:-1]
            target = all_traj[-1]

            target_tim = train_data[user]['sessions'][traj][-1][1]
            target_cat = train_data[user]['sessions'][traj][-1][-1]

            num_poi = len(set(hist_traj))

            poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                     index=set(hist_traj),
                                     columns=set(hist_traj))
            dist_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                      index=set(hist_traj),
                                      columns=set(hist_traj))
            current_time = copy.deepcopy([poi[1] for poi in train_data[user]['sessions'][traj]][:-1])
            current_cat = [cat[-1] for cat in train_data[user]['sessions'][traj][:-1]]
            current_latlon = [[latlon[3],latlon[2]] for latlon in train_data[user]['sessions'][traj][:-1]]
            temp_time = {}
            temp_cat = {}
            temp_latlon = {}
            tensor_tim_list = {}
            tensor_cat_list = {}
            tensor_latlon_list = {}
            for i in range(len(hist_traj)):
                if i > 0:
                    poi_graph.loc[hist_traj[i - 1], hist_traj[i]] += 1
                temp_time[hist_traj[i]] = copy.deepcopy(current_time[i])
                temp_cat[hist_traj[i]] = copy.deepcopy(current_cat[i])
                temp_latlon[hist_traj[i]] = copy.deepcopy(current_latlon[i])
            df = pd.DataFrame(hist_traj)
            df = df.drop_duplicates(keep='last')
            newtraj_list = df.to_numpy().reshape(len(df.to_numpy())).tolist()
            for i in temp_time.keys():
                tensor_tim_list[i] = copy.deepcopy(temp_time[i])
            for i in temp_cat.keys():
                tensor_cat_list[i] = copy.deepcopy(temp_cat[i])
            for i in temp_latlon.keys():
                tensor_latlon_list[i] = copy.deepcopy(temp_latlon[i])
            newtensor_tim_list = []
            newtensor_tim_normal_list = []
            newtensor_cat_list = []
            newtensor_latlon_list = []
            newtensor_distance_list = []
            for i in range(len(tensor_tim_list)):
                newtensor_tim_list.append(tensor_tim_list[newtraj_list[i]])
                if tensor_tim_list[newtraj_list[i]]==0:
                # if tensor_tim_list[newtraj_list[i]] == 1:
                    newtensor_tim_normal_list.append(0)
                else:
                    newtensor_tim_normal_list.append((tensor_tim_list[newtraj_list[i]]) / 48)
                    # newtensor_tim_normal_list.append((tensor_tim_list[newtraj_list[i]]-1)/48)
            for i in range(len(tensor_cat_list)):
                newtensor_cat_list.append(tensor_cat_list[newtraj_list[i]])
            for i in range(len(tensor_latlon_list)):
                newtensor_latlon_list.append(tensor_latlon_list[newtraj_list[i]])
                if i < len(tensor_latlon_list)-1:
                    newtensor_distance_list.append(LLs2Dist(tensor_latlon_list[newtraj_list[i]][0],tensor_latlon_list[newtraj_list[i]][1],tensor_latlon_list[newtraj_list[i+1]][0],tensor_latlon_list[newtraj_list[i+1]][1]))
            poi_graph = poi_graph[newtraj_list]
            poi_graph = poi_graph.reindex(newtraj_list)

            dict_data[user][traj]['num_node'] = len(set(hist_traj))
            dict_data[user][traj]['node_name'] = copy.deepcopy(torch.LongTensor(newtraj_list))
            dict_data[user][traj]['edge_type'] = copy.deepcopy(torch.LongTensor(poi_graph.values))
            # dict_data[user][traj]['edge_type'] = copy.deepcopy(torch.FloatTensor((poi_graph+dist_graph).values))
            dict_data[user][traj]['target'] = copy.deepcopy(torch.LongTensor([target]))
            dict_data[user][traj]['target_tim'] = copy.deepcopy(torch.LongTensor([target_tim]))
            dict_data[user][traj]['target_cat'] = copy.deepcopy(torch.LongTensor([target_cat]))
            dict_data[user][traj]['time']  = copy.deepcopy(torch.LongTensor(newtensor_tim_list))
            dict_data[user][traj]['time_normal'] = copy.deepcopy(torch.FloatTensor(newtensor_tim_normal_list))
            dict_data[user][traj]['user'] = copy.deepcopy(torch.LongTensor([user]))
            dict_data[user][traj]['cat'] = copy.deepcopy(torch.LongTensor(newtensor_cat_list))
            dict_data[user][traj]['latlon'] = copy.deepcopy(torch.LongTensor(newtensor_cat_list))
            dict_data[user][traj]['distance'] = copy.deepcopy(torch.LongTensor(newtensor_cat_list))
    return dict_data

def gen_poigraph_d0103_nyc_getnext(train_data):
    train_df=[]
    test_df=[]
    val_df=[]
    for i in train_data.keys():
        for j in train_data[i]['sessions'].keys():
            if j in train_data[i]['train']:
                for k in range(len(train_data[i]['sessions'][j])):
                    # if train_data[i]['sessions'][j][k][1]==1:
                    if train_data[i]['sessions'][j][k][1] == 0:
                        train_df.append([i, train_data[i]['sessions'][j][k][0], train_data[i]['sessions'][j][k][1],
                                         str(i) + '_' + str(j)])
                        # train_df.append([i, train_data[i]['sessions'][j][k][0], train_data[i]['sessions'][j][k][1]-1,
                        #                  str(i) + '_' + str(j)])
                    else:
                        train_df.append(
                            [i, train_data[i]['sessions'][j][k][0], (train_data[i]['sessions'][j][k][1]) / 48,
                             str(i) + '_' + str(j)])
                        # train_df.append([i, train_data[i]['sessions'][j][k][0], (train_data[i]['sessions'][j][k][1]-1)/48, str(i)+'_'+str(j) ])
            else:
                for k in range(len(train_data[i]['sessions'][j])):
                    # if train_data[i]['sessions'][j][k][1] == 1:
                    if train_data[i]['sessions'][j][k][1] == 0:
                        val_df.append(
                            [i, train_data[i]['sessions'][j][k][0], train_data[i]['sessions'][j][k][1],
                             str(i) + '_' + str(j)])
                        # val_df.append(
                        #     [i, train_data[i]['sessions'][j][k][0], train_data[i]['sessions'][j][k][1] - 1,
                        #      str(i) + '_' + str(j)])
                    else:
                        val_df.append(
                            [i, train_data[i]['sessions'][j][k][0], (train_data[i]['sessions'][j][k][1]) / 48,
                             str(i) + '_' + str(j)])
                        # val_df.append([i, train_data[i]['sessions'][j][k][0], (train_data[i]['sessions'][j][k][1]-1)/48, str(i)+'_'+str(j)])
    test_df=copy.deepcopy(val_df)
    return train_df, val_df, test_df

def run(args):
    folder = ""
    if args.dataset_name == "foursquare_tky":
        folder = "foursquare_tky"
    elif args.dataset_name == "toyota":
        folder = "toyota"
    elif args.dataset_name == "gowalla_nevda":
        folder = "gowalla_nevda"
        
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.dataset_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path)
    print('model_mode: {} | history_mode:{} | users:{}'.format(
        parameters.model_mode, parameters.history_mode, parameters.uid_size))

    # ---- Generation used in deepmove paper
    # candidate = parameters.data_neural.keys()
    # data_train, train_idx = generate_input_long_history_graph(parameters.data_neural, 'train', candidate=candidate)
    # data_test, test_idx = generate_input_long_history_graph(parameters.data_neural, 'test', candidate=candidate)
    
    # data_train, train_idx = generate_input_history(parameters.data_neural, 'train', mode2=parameters.history_mode,
    #                                                    candidate=candidate)
    # data_test, test_idx = generate_input_history(parameters.data_neural, 'test', mode2=parameters.history_mode,
    #                                                 candidate=candidate)


    # ----- Generation for the MobGT
    print("--- Pickle files generation ---")
    gen_train = gen_poigraph_d1228_nyc_avg_maxtime(parameters.data_neural, 'train')
    # gen_train = gen_poigraph_d1212_nyc_avg_maxtime(data_train, 'train')

    gen_test = gen_poigraph_d1228_nyc_avg_maxtime(parameters.data_neural, 'test')
    # gen_test = gen_poigraph_d1212_nyc_avg_maxtime(data_test, 'test')

    train_idx = {}
    for user in gen_train.keys():
        train_idx[user] = []
        for i in gen_train[user].keys():
            train_idx[user].append(i)
            
    test_idx = {}
    for user in gen_test.keys():
        test_idx[user]=[]
        for i in gen_test[user].keys():
            test_idx[user].append(i)

    with open('../dataset/{}/raw/train.pickle'.format(folder), 'wb') as fw:
        pickle.dump(gen_train, fw)
    
    with open('../dataset/{}/raw/test.pickle'.format(folder), 'wb') as fw:
        pickle.dump(gen_test, fw)

    with open('../dataset/{}/raw/train_idx.pkl'.format(folder), 'wb') as fw:
        pickle.dump(train_idx, fw)

    with open('../dataset/{}/raw/test_idx.pkl'.format(folder), 'wb') as fw:
        pickle.dump(test_idx, fw)


def load_pretrained_model(config):
    res = json.load(open("../pretrain/" + config.model_mode + "/res.txt"))
    args = Settings(config, res["args"])
    return args


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["dataset_name"]
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
    parser.add_argument('--dataset_name', type=str, default='foursquare_tky')
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
    parser.add_argument('--data_path', type=str, default='../dataset/pickle_files/')
    parser.add_argument('--save_path', type=str, default='../results/')
    parser.add_argument('--model_mode', type=str, default='attn_local_long',
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()
    if args.pretrain == 1:
        args = load_pretrained_model(args)

    ours_acc = run(args)
