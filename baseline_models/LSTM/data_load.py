import copy

import torch
from torch.autograd import Variable

import numpy as np
import pickle as pickle
import pandas as pd
import copy
from collections import deque, Counter

class PoiEmbedding():

    def __init__(self, data_neural,train_data, poi_num):

        # self.dis_matric = dis_matric
        # self.walk_num = walk_num
        # self.walk_length = walk_length
        self.train_data = train_data
        self.poi_num = poi_num
        # self.sentences = []
        self.positive_data = []
        self.data_neural=data_neural
        self.dataset = None

    def num_poi(self):
        index=len(self.data_neural)
        poi=[]
        for i in range(index):
            session_index=len(self.data_neural[i]['sessions'])
            for j in range(session_index):
                if poi==[]:
                    poi=copy.deepcopy(self.data_neural[i]['sessions'][j])
                else:
                    poi.extend(self.data_neural[i]['sessions'][j])
        poi=np.array(poi).reshape(len(poi)*2).tolist()[::2] #删掉时间
        num=len(set(poi))
        return num


    def gen_train(self):
        dict_data={}
        self.user_poigraph=copy.deepcopy(self.train_data)
        for user in self.train_data:
            dict_data[user]={}
            # index=len(data_train[user])
            for traj in self.train_data[user]:
                dict_data[user][traj]={}
                for current_traj in self.train_data[user][traj]['history_loc']:
                    hist_traj = data_train[user][traj]['history_loc'].numpy()
                    hist_traj = traj.reshape(len(traj)).tolist()
                    num_poi = len(set(hist_traj))
                    self.poi_graph = pd.DataFrame(np.zeros(len(num_poi) ** 2).reshape(num_poi, num_poi),
                                              index=set(traj),
                                              columns=set(traj))
                    if(traj[i+1] == 0):
                        break
                    self.poi_graph.loc[traj[i],traj[i+1]] += 1
        # self.poi_graph = self.poi_graph + self.dis_matric

        for i in self.poi_graph.index:
            if np.sum(self.poi_graph.loc[i]) == 0:
                continue
            self.poi_graph.loc[i] = self.poi_graph.loc[i] / np.sum(self.poi_graph.loc[i])
        print(self.poi_graph)

def load(data_path,data_name):
    data = pickle.load(open(data_path + data_name + '.pk', 'rb'),encoding='latin1')
    return data

def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i] #取每个session(第一个session没取，被上面的if判断跳过了)
            target = np.array([s[0] for s in session[1:]]) #一个session有11个POI，这就相当于取了前10个

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])

            history_tim = [t[1] for t in history] #取时间
            history_count = [1]
            last_t = history_tim[0]
            count = 1
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
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]]) #看起来是把每次的session加上形成一个大序列
            #又是一个逐一分割操作
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target))
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx



if __name__ == '__main__':
    data_path="../data/"
    data_name="toyota"
    data=load(data_path,data_name)
    vid_list = data['vid_list']
    uid_list = data['uid_list']
    data_neural = data['data_neural']
    candidate = data_neural.keys()
    data_train,train_idx=generate_input_long_history(data_neural,'train',candidate)
