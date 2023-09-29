# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable

import numpy as np
import pickle as pickle 
import pandas as pd
import math 
import copy
from collections import deque, Counter


class RnnParameterData(object):
    def __init__(self, loc_emb_size=500, uid_emb_size=40, voc_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                 history_mode='avg', attn_type='dot', epoch_max=30, rnn_type='LSTM', model_mode="simple",
                 data_path='../dataset/', save_path='../results/', data_name='toyota1'):#foursquare_cut_one_day
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        data = pickle.load(open(self.data_path + self.data_name + '.pk', 'rb'),encoding='latin1')
        self.vid_list = data['vid_list']
        self.uid_list = data['uid_list']
        self.data_neural = data['data_neural']

        self.tim_size = 48
        self.loc_size = len(self.vid_list)
        self.uid_size = len(self.uid_list)
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.voc_emb_size = voc_emb_size
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.history_mode = history_mode
        self.model_mode = model_mode


def generate_input_history(data_neural, mode, mode2=None, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            # voc_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), 27))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            # trace['voc'] = Variable(torch.LongTensor(voc_np))

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)

            # merge traces with same time stamp
            if mode2 == 'max':
                history_tmp = {}
                for tr in history:
                    if tr[1] not in history_tmp:
                        history_tmp[tr[1]] = [tr[0]]
                    else:
                        history_tmp[tr[1]].append(tr[0])
                history_filter = []
                for t in history_tmp:
                    if len(history_tmp[t]) == 1:
                        history_filter.append((history_tmp[t][0], t))
                    else:
                        tmp = Counter(history_tmp[t]).most_common()
                        if tmp[0][1] > 1:
                            history_filter.append((history_tmp[t][0], t))
                        else:
                            ti = np.random.randint(len(tmp))
                            history_filter.append((tmp[ti][0], t))
                history = history_filter
                history = sorted(history, key=lambda x: x[1], reverse=False)
            elif mode2 == 'avg':
                history_tim = [t[1] for t in history]
                history_count = [1]
                last_t = history_tim[0]
                count = 1
                for t in history_tim[1:]:
                    if t == last_t:
                        count += 1
                    else:
                        history_count[-1] = count
                        history_count.append(1)
                        last_t = t
                        count = 1
            ################

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            if mode2 == 'avg':
                trace['history_count'] = history_count

            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_input_long_history2(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}

        trace = {}
        session = []
        for c, i in enumerate(train_id):
            session.extend(sessions[i])
        target = np.array([s[0] for s in session[1:]])

        loc_tim = []
        loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
        loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
        tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
        trace['loc'] = Variable(torch.LongTensor(loc_np))
        trace['tim'] = Variable(torch.LongTensor(tim_np))
        trace['target'] = Variable(torch.LongTensor(target))
        data_train[u][i] = trace
        # train_idx[u] = train_id
        if mode == 'train':
            train_idx[u] = [0, i]
        else:
            train_idx[u] = [i]
    return data_train, train_idx


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
            latlon=[]
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
                    latlon.extend([(s[2], s[3]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
                latlon.extend([(s[2], s[3]) for s in sessions[train_id[j]]])

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

            latlon.extend([(s[2], s[3]) for s in session[:-1]])

            #又是一个逐一分割操作
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            latlon_np = np.reshape(np.array([s for s in latlon]), (len(latlon), 2))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['latlon'] = Variable(torch.FloatTensor(latlon_np))
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = list(train_idx.keys()) #获取所有用户ID
    train_queue = deque()
    np.random.seed(43)
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:]) #为什么从1开始取捏？
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user) #每次都打乱一下用户顺序，然后按照下面的方法出至多8个
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)): #为什么大于等于int(8.86)停止?
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0]) #有点像在检测initial_queue是否出完了
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue #这其实把所有用户的数据混一起了用户之间没有前后关系，用户内部有前后关系


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


def get_hint(target, scores, users_visited):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(1, 1)
    predx = idxx.cpu().numpy()
    hint = np.zeros((3,))
    count = np.zeros((3,))
    count[0] = len(target)
    for i, p in enumerate(predx):
        t = target[i]
        if t == p[0] and t > 0:
            hint[0] += 1
        if t in users_visited:
            count[1] += 1
            if t == p[0] and t > 0:
                hint[1] += 1
        else:
            count[2] += 1
            if t == p[0] and t > 0:
                hint[2] += 1
    return hint, count


def run_simple(data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2=None):
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    run_queue = None
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)

    users_acc = {}
    for c in range(queue_len):
        optimizer.zero_grad()
        u, i = run_queue.popleft() #从总序列开始出队列了
        if u not in users_acc:
            users_acc[u] = [0, 0, 0]
        loc = data[u][i]['loc'].cuda() #其实是用户第一个会话和第二个会话的合体
        tim = data[u][i]['tim'].cuda()
        target = data[u][i]['target'].cuda()
        uid = Variable(torch.LongTensor([u])).cuda() #把用户ID也输入进去

        #如果注意到会话之间的数量为什么不都是11而有区别的时候，其原因是由于72小时包含的数据量的问题。有的72个小时包含的数据点不到11个，所以他的数量就少

        if 'attn' in mode2:
            history_loc = data[u][i]['history_loc'].cuda()
            history_tim = data[u][i]['history_tim'].cuda()

        if mode2 in ['simple', 'simple_long']:
            scores = model(loc, tim)
        elif mode2 == 'attn_avg_long_user':
            history_count = data[u][i]['history_count']
            target_len = target.data.size()[0]
            scores = model(loc, tim, history_loc, history_tim, history_count, uid, target_len)
        elif mode2 == 'attn_local_long':
            target_len = target.data.size()[0]
            scores = model(loc, tim, target_len)

        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]
        loss = criterion(scores, target)

        if mode == 'train':
            loss.backward()
            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                #pytorch中梯度剪裁方法为 torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2)
                # 1。三个参数：parameters：希望实施梯度裁剪的可迭代网络参数
                #max_norm：该组网络参数梯度的范数上限norm_type：范数类型
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
        elif mode == 'test':
            users_acc[u][0] += len(target)
            acc = get_acc(target, scores)
            users_acc[u][1] += acc[2]
            users_acc[u][2] += acc[1]
        total_loss.append(loss.data.cpu().numpy())

    avg_loss = np.mean(total_loss, dtype=np.float64)
    tmp_acc = [0.0,0.0]
    if mode == 'train':
        return model, avg_loss
    elif mode == 'test':
        users_rnn_acc = {}
        # for u in users_acc:
        #     tmp_acc = users_acc[u][1] / users_acc[u][0]
        #     users_rnn_acc[u] = tmp_acc.tolist()[0]
        # avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])
        # return avg_loss, avg_acc, users_rnn_acc
        sum_test_samples = 0.0
        for u in users_acc:
            tmp_acc[0] = users_acc[u][1] + tmp_acc[0]
            tmp_acc[1] = users_acc[u][2] + tmp_acc[1]
            sum_test_samples = sum_test_samples + users_acc[u][0]
            tmps_acc = users_acc[u][1] / users_acc[u][0]
            users_rnn_acc[u] = tmps_acc.tolist()[0]
        avg_acc = (np.array(tmp_acc)/sum_test_samples).tolist()
        return avg_loss, avg_acc, users_rnn_acc


def markov(parameters, candidate):
    validation = {}
    for u in candidate:
        traces = parameters.data_neural[u]['sessions'] #取一个用户的会话
        train_id = parameters.data_neural[u]['train'] #取训练数据
        test_id = parameters.data_neural[u]['test'] #取测试数据
        #以第一个用户为例（考虑起点终点的数据集），总共有25个会话，前20个用来训练，后5个用来测试
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]]) #取POI ID
        locations_train = []
        for t in trace_train:
            locations_train.extend(t) #把所有的合并到一起
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])
        locations_test = []
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test] #记录了每个用户的训练和测试集的POI ID
    acc = 0
    acc5=0
    count = 0
    user_acc = {}
    user5_acc={}
    for u in validation.keys():
        topk = list(set(validation[u][0])) #类似于统计有多少不同种的POI的，并且按升序排列
        transfer = np.zeros((len(topk), len(topk)))

        # train
        sessions = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0] #当前POI
                target = sessions[i][j + 1][0] #下一个POI
                if loc in topk and target in topk: #还会有不存在的场景吗？
                    r = topk.index(loc)
                    c = topk.index(target)
                    transfer[r, c] += 1
        for i in range(len(topk)):
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum #归一化了，用次数算频率

        # validation 测试部分
        user_count = 0
        user_acc[u] = 0
        user5_acc[u] = 0
        test_id = parameters.data_neural[u]['test']
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                count += 1
                user_count += 1
                if loc in topk:
                    pred = np.argmax(transfer[topk.index(loc), :]) #输出概率最大的那个
                    pred5=(-transfer[topk.index(loc),:]).argsort()[:5] #输出前五个
                    if pred >= len(topk) - 1:
                        pred = np.random.randint(len(topk))
                    #这些部分都是命中了就+1的意思，就是统计命中了多少次的
                    pred2 = topk[pred]
                    if pred2 == target:
                        acc += 1
                        user_acc[u] += 1
                    for k in range(len(pred5)):
                        top5=topk[pred5[k]]
                        if top5==target:
                            acc5+=1
                            user5_acc[u]+=1
                            break
                        else:
                            continue

        user_acc[u] = user_acc[u] / user_count
        user5_acc[u] = user5_acc[u] / user_count
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    avg5_acc = np.mean([user5_acc[u] for u in user5_acc])
    return avg_acc, user_acc,avg5_acc

def LLs2Dist(lat1, lon1, lat2, lon2):
        R = 6371
        dLat = (lat2 - lat1) * math.pi / 180.0
        dLon = (lon2 - lon1) * math.pi / 180.0

        a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(lat1 * math.pi / 180.0) * math.cos(lat2 * math.pi / 180.0) * math.sin(dLon / 2) * math.sin(dLon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist = R * c
        return dist
    
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
                        dist_graph.loc[hist_traj[i-1], hist_traj[i]]=dist
                dict_data[user][traj]['num_node']=len(set(hist_traj))
                dict_data[user][traj]['node_name']=copy.deepcopy(torch.LongTensor(list(set(hist_traj))))
                dict_data[user][traj]['edge_type']=copy.deepcopy(torch.LongTensor(list(poi_graph)))
                dict_data[user][traj]['dist_type'] = copy.deepcopy(torch.FloatTensor(list(dist_graph)))
                # dict_data[user][traj]['normalized']=0
                dict_data[user][traj]['time'] = copy.deepcopy(torch.LongTensor(time))

                # target未解决
                if traj==len(train_data[user]):
                    hist_traj = train_data[user][traj]['loc'].numpy()
                    hist_traj = hist_traj.reshape(len(hist_traj)).tolist()
                    num_poi = len(set(hist_traj))
                    time = train_data[user][traj]['tim'][0:len(hist_traj)]
                    poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                            index=set(hist_traj),
                                            columns=set(hist_traj))
                    for i in range(len(hist_traj)):
                        if i > 0:
                            poi_graph.loc[hist_traj[i - 1], hist_traj[i]] += 1
                            latlon1 = train_data[user][traj]['latlon'][i - 1]
                            latlon2 = train_data[user][traj]['latlon'][i]
                            dist = LLs2Dist(latlon1[0], latlon1[1], latlon2[0], latlon2[1])
                            dist_graph.loc[hist_traj[i - 1], hist_traj[i]] = dist
                    dict_data[user][traj]['num_node'] = len(set(hist_traj))
                    dict_data[user][traj]['node_name']=copy.deepcopy(torch.LongTensor(list(set(hist_traj))))
                    dict_data[user][traj]['edge_type']=copy.deepcopy(torch.LongTensor(list(poi_graph)))
                    dict_data[user][traj]['dist_type'] = copy.deepcopy(torch.FloatTensor(list(dist_graph)))
                    # dict_data[user][traj]['normalized'] = 0
                    dict_data[user][traj]['time'] = copy.deepcopy(torch.LongTensor(time))
        return dict_data

