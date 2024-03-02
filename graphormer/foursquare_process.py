from __future__ import print_function
from __future__ import division

import time
import argparse
import numpy as np
import pickle
import pandas as pd
import math
from collections import Counter
from math import radians, cos, sin, asin, sqrt
import itertools
from tqdm import tqdm

def LLs2Dist(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(lat1 * math.pi / 180.0) * math.cos(lat2 * math.pi / 180.0) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist = R * c
    return dist

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance

def entropy_spatial(sessions):
    locations = {} #喂进来了所有session
    days = sorted(sessions.keys())
    for d in days:
        session = sessions[d]
        for s in session:
            if s[0] not in locations:
                locations[s[0]] = 1
            else:
                locations[s[0]] += 1
    frequency = np.array([locations[loc] for loc in locations])
    frequency = frequency / np.sum(frequency)
    entropy = - np.sum(frequency * np.log(frequency))
    return entropy

def timengtonum(timeeng):
    timdict=dict(Jan='01', Feb='02', Mar='03', Apr='04', May='05', Jun='06', Jul='07', Aug='08', Sep='09', Oct='10', Nov='11', Dec='12')
    return timdict[timeeng]

class DataFoursquare(object):
    def __init__(self, trace_min=10, global_visit=10, hour_gap=24, min_gap=10, session_min=5, session_max=10,
                 sessions_min=2, train_split=0.8, embedding_len=50):
    # def __init__(self, trace_min=10, global_visit=10, hour_gap=0, min_gap=0, session_min=2, session_max=10,
    #                  sessions_min=2, train_split=0.8, embedding_len=50):
        tmp_path = "../dataset/original_txt_data/"
        self.TWITTER_PATH = tmp_path + 'dataset_TSMC2014_TKY.txt'
        self.SAVE_PATH = tmp_path
        self.save_name = 'foursquare_tky'

        self.trace_len_min = trace_min
        self.location_global_visit_min = global_visit
        self.hour_gap = hour_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min
        self.sessions_count_min = sessions_min
        self.words_embeddings_len = embedding_len

        self.train_split = train_split

        self.data = {}
        self.venues = {}
        self.words_original = []
        self.words_lens = []
        self.dictionary = dict()
        self.words_dict = None
        self.data_filter = {}
        self.user_filter3 = None
        self.uid_list = {}
        self.vid_list = {'unk': [0, -1]}
        self.vid_list_lookup = {}
        self.vid_lookup = {}
        self.pid_loc_lat = {}
        self.data_neural = {}

        self.pid_pname = {}
        self.vname_lookup={}

        self.catid_list = {'unk': [0, -1]}
        self.catid_list_lookup = {}
        self.catid_lookup = {}

        self.vid_list_train = {'unk': [0, -1]}
        self.vid_list_lookup_train = {}
        self.vid_lookup_train = {}

        self.catid_list_train = {'unk': [0, -1]}
        self.catid_list_lookup_train = {}
        self.catid_lookup_train = {}

        # self.catid={}
    # ############# 1. read trajectory data from twitters
    @staticmethod
    def tid_list_unix(tmd):
        tm = time.strptime(tmd,
                           "%Y-%m-%d %H:%M:%S")  # time.struct_time(tm_year=2010, tm_mon=6, tm_mday=5, tm_hour=0, tm_min=11, tm_sec=12, tm_wday=5, tm_yday=156, tm_isdst=-1)
        tid = int(time.mktime(tm))
        return tid



    def load_trajectory_from_tweets(self):
        with open(self.TWITTER_PATH, errors='ignore',encoding='utf-8') as fid:
            for i, line in enumerate(fid):
                # _, uid, lat, lon, tim, _, _, tweet, pid = line.strip('\r\n').split('')# 抽取出userid,time,poi id,tweet没被用上
                # tim1, tim2, _, _, uid, pid = line.strip('\r\n').split(' ')
                # tim1, tim2, lat, lon, uid, pid = line.strip('\r\n').split(' ')
                # uid, tim, lat, lon, pid = line.strip('\r\n').split(',')
                # uid, tim, lat, lon, pid, pname = line.strip('\r\n').split(',')
                # pname=pname.replace(u'\u3000', ' ')
                # _, tim1=tim1.split('"')
                # tim2,_ = tim2.split('"')
                # tim=tim1+' '+tim2
                
                #for Yangs' Foursquare Data
                #---------------------------------------------------------------------------------
                uid, pid, cat_id, cat_name, lat, lon, offset, tim = line.strip('\r\n').split('\t')
                tim_len = len(tim)
                tim = tim[tim_len - 4:tim_len] + '-' + timengtonum(tim[4:7]) + '-' + tim[8:10] + tim[10:19]
                tim = time.localtime(int(time.mktime(time.strptime(tim, "%Y-%m-%d %H:%M:%S")))++int(offset)*60)
                tim = time.strftime("%Y-%m-%d %H:%M:%S", tim)
                # ---------------------------------------------------------
                # uid, tim, tim_slots, coords, pid= line.strip('\r\n').split('\t')
                # tim=tim[:10]+' '+tim[11:-1]

                #for Toyota Data
                #------------------------------------------------------------------
                # uid, tim, lat, lon, pid, pname = line.strip('\r\n').split(',')
                # uid, pid, cat_id, cat_name, lat, lon, offset, tim = line.strip('\r\n').split('\t')
                # cat_id=pid[4:6]
                #-----------------------------------------------------------------------------------

                #for Gowalla
                #-----------------------------------------
                # if i ==0:
                #     continue
                # # uid, pid, _, cat_id, _, lat, lon, _, tim,=line.strip('\r\n').split(',')
                # uid, pid, tim, lat, lon, cat_id = line.strip('\r\n').split(',')
                #------------------------------------------

                # for GETNExt
                # -----------------------------------------
                # if i == 0:
                #     continue
                # uid, pid, tim, lat, lon, cat_id = line.strip('\r\n').split(',')
                # ------------------------------------------

                if uid not in self.data:
                    # self.data[uid] = [[pid, tim]]
                    self.data[uid] = [[pid, tim, lat, lon, cat_id]]
                    # self.data[uid] = [[pid,tim,lat,lon,pname]]
                else:
                    # self.data[uid].append([pid, tim])
                    self.data[uid].append([pid, tim, lat, lon, cat_id])
                    # self.data[uid].append([pid,tim,lat,lon,pname])
                if pid not in self.venues:
                    self.venues[pid] = 1
                else:
                    self.venues[pid] += 1
                # if cat_id not in self.catid:
                #     self.catid[cat_id] = 1
                # else:
                #     self.catid[cat_id] += 1
            print("--- finished executing load_trajectory_from_tweets ---")
            print("data uid: ", list(itertools.islice(self.data.items(), 0, 1))[0][0])
            print("data [pid, tim]: ", list(itertools.islice(self.data.items(), 0, 1))[0][1][0])
            print("venues: ", list(itertools.islice(self.venues.items(), 0, 4)))

    # ########### 3.0 basically filter users based on visit length and other statistics
    def filter_users_by_length(self):
        uid_3 = [x for x in self.data if len(self.data[x]) >= self.trace_len_min] #用户的签到次数少于trace_len_min时被舍弃
        pick3 = sorted([(x, len(self.data[x])) for x in uid_3], key=lambda x: x[1], reverse=True) #统计每个用户的签到次数按降序排列
        pid_3 = [x for x in self.venues if self.venues[x] >= self.location_global_visit_min] #这个是POI集合总数43380, POI出现的次数少于location_global_visit_min时被舍弃
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True) #统计POI出现次数按降序排列
        pid_3 = dict(pid_pic3)

        session_len_list = []
        for u in pick3:
            uid = u[0]
            info = self.data[uid] #取当前用户的签到信息
            topk = Counter([x[0] for x in info]).most_common() #计算user u[0]的所有签到的POI的出现频率并返回元组
            topk1 = [x[0] for x in topk if x[1] > 1] #删除掉只出现一次的POI
            sessions = {}
            for i, record in enumerate(info):
                poi, tim, lat, lon, cat_id = record
                try:
                    tid = int(time.mktime(time.strptime(tim, "%Y-%m-%d %H:%M:%S"))) #时间转化为数字,差1就是差1秒
                    tmd_day=time.strptime(tim, "%Y-%m-%d %H:%M:%S").tm_mday

                    #for Gowalla
                    # tid = int(time.mktime(time.strptime('1970-01-02 '+tmd, "%Y-%m-%d %H:%M")))
                except Exception as e:
                    print('error:{}'.format(e))
                    continue
                sid = len(sessions)

                # if poi not in pid_3 and poi not in topk1:
                if poi not in pid_3:
                # if poi not in pid_3 or poi not in topk1:#用户的签到数据不在POI数据集里面并且签到数据不在只出现一次的POI里面（防止topk1里面的POI不在pid_3里面）
                    # if poi not in topk1:
                    continue

                # if i == 0 or len(sessions) == 0:
                #     sessions[sid] = [record]
                #     last_tid=tid
                # else:
                #     if tid-last_tid>(60*60*24*7):
                #         sessions[sid] = [record]  # 这个其实代表新开了一个会话id
                #         last_tid = tid
                #     else:
                #         sessions[sid - 1].append(record)




                #-----------------------------------------------------------原有部分
                if i == 0 or len(sessions) == 0:
                    # sessions[sid] = [[record[0],self.tid_list_unix(record[1])]]
                    sessions[sid] = [record]
                    # last_day = tmd_day
                    # last_tid = tid
                else:
                    if (tid - last_tid) / 3600 >= self.hour_gap or len(sessions[sid - 1]) > self.session_max: #距离上次签到大于小时级别的阈值(72*1h)或者会话的长度到了，会话的长度就是10
                    # if tmd_day!=last_day:
                        sessions[sid] = [record] #这个其实代表新开了一个会话id
                        # last_day = tmd_day
                        # last_tid = tid
                    elif (tid - last_tid) / 60 >= self.min_gap: #必须距离上次签到大于分钟级别的阈值（self.min_gap=10->10min）
                    # else:
                        sessions[sid - 1].append(record) #如果满足这个条件就增加一组数据到上一个会话id中


                last_tid = tid

            # ------------------------------------------------
            sessions_filter = {}
            for s in sessions:
                if len(sessions[s]) >= self.filter_short_session: #删掉不到五个签到点的会话
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s]))
            if len(sessions_filter) >= self.sessions_count_min: #不是很懂为什么要这句话, 感觉效果是一样的啊
                self.data_filter[uid] = {'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                                         'sessions': sessions_filter, 'raw_sessions': sessions} #len(sessions_filter)代表了有多少会话,len(topk)记录了用户访问的POI数量（非次数，数量指的更像是种类）

        self.user_filter3 = [x for x in self.data_filter if
                             self.data_filter[x]['sessions_count'] >= self.sessions_count_min] #用户如果少于5个的会话则舍弃这个用户

    # ########### 4. build dictionary for users and location
    def build_users_locations_dict(self):
        # self.user_filter3=self.user_filter3[0:100]
        for u in self.user_filter3:
            sessions = self.data_filter[u]['sessions']
            # sessions = {x: sessions[x] for x in range(int(len(sessions)*0.1))}
            if u not in self.uid_list:
                self.uid_list[u] = [len(self.uid_list), len(sessions)] #记录下每个用户的会话数和他在uid_list中的位置(index)
            for sid in sessions:
                #对于定长的graphormer
                poi = [p[0] for p in sessions[sid]] #提取所有POI
                lat = [p[2] for p in sessions[sid]]
                lon = [p[3] for p in sessions[sid]]
                cat = [p[-1] for p in sessions[sid]]
                index=0
                for p in poi:
                    if p not in self.vid_list:       #self.vid_list = {'unk': [0, -1]} 又是统计POI出现的次数并给他一个编号，而且编号共同，所有用户同享
                        self.vid_list_lookup[len(self.vid_list)] = p
                        self.vid_list[p] = [len(self.vid_list), 1, lat[index], lon[index], cat[index], 0]
                        temp = len(self.vid_list[p])-2
                    else:
                        self.vid_list[p][1] += 1
                    index+=1
                for c in cat:
                    if c not in self.catid_list:  # self.vid_list = {'unk': [0, -1]} 又是统计POI出现的次数并给他一个编号，而且编号共同，所有用户同享
                        self.catid_list_lookup[len(self.catid_list)] = c
                        self.catid_list[c] = [len(self.catid_list), 1]
                    else:
                        self.catid_list[c][1] += 1
        for i in self.vid_list:
            if i =='unk':
                continue
            self.vid_list[i][-1] = self.catid_list[self.vid_list[i][temp]][1]
            self.vid_list[i][temp]=self.catid_list[self.vid_list[i][temp]][0]

    # support for radius of gyration
    def load_venues(self):
        with open(self.TWITTER_PATH, errors='ignore',encoding='utf-8') as fid:
            for line in fid:
                # _, uid, lon, lat, tim, _, _, tweet, pid = line.strip('\r\n').split('')
                # tim1,tim2, lon, lat, uid, pid = line.strip('\r\n').split(' ')
                # uid, tim, lat, lon, pid = line.strip('\r\n').split(',')
                # uid, tim, lat, lon, pid, pname = line.strip('\r\n').split(',')
                uid, pid, cat_id, cat_name, lat, lon, offset, tim = line.strip('\r\n').split('\t')
                cat_name = cat_name.replace(u'\u3000', ' ')
                # _, tim1 = tim1.split('"')
                # tim2, _ = tim2.split('"')
                # tim = tim1 + ' ' + tim2
                self.pid_loc_lat[pid] = [float(lon), float(lat)]
                self.pid_pname[pid]=cat_name


    def venues_lookup(self):
        for vid in self.vid_list_lookup:
            pid = self.vid_list_lookup[vid]
            lon_lat = self.pid_loc_lat[pid]
            # pname = self.pid_pname[pid]#这不是脱裤子放屁吗
            self.vid_lookup[vid] = lon_lat
            # self.vname_lookup[vid]=pname


    # ########## 5.0 prepare training data for neural network
    @staticmethod
    def tid_list(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        tid = tm.tm_wday * 24 + tm.tm_hour
        return tid

    @staticmethod
    def tid_list_48(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S") #time.struct_time(tm_year=2010, tm_mon=6, tm_mday=5, tm_hour=0, tm_min=11, tm_sec=12, tm_wday=5, tm_yday=156, tm_isdst=-1)
        if tm.tm_wday in [0, 1, 2, 3, 4]: #判断是不是工作日
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return tid

    @staticmethod
    def tid_list_48_1(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S") #time.struct_time(tm_year=2010, tm_mon=6, tm_mday=5, tm_hour=0, tm_min=11, tm_sec=12, tm_wday=5, tm_yday=156, tm_isdst=-1)
        if tm.tm_wday in [0, 1, 2, 3, 4]: #判断是不是工作日
            tid = tm.tm_hour+1
        else:
            tid = tm.tm_hour + 25
        return tid

    @staticmethod
    def tid_list_7days(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")  # time.struct_time(tm_year=2010, tm_mon=6, tm_mday=5, tm_hour=0, tm_min=11, tm_sec=12, tm_wday=5, tm_yday=156, tm_isdst=-1)
        tid = tm.tm_hour+(tm.tm_wday*24)+1
        return tid

    @staticmethod
    def tid_list_1day48(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S") #time.struct_time(tm_year=2010, tm_mon=6, tm_mday=5, tm_hour=0, tm_min=11, tm_sec=12, tm_wday=5, tm_yday=156, tm_isdst=-1)
        if tm.tm_min<30:
            tid= tm.tm_hour*2
        else:
            tid= tm.tm_hour*2 + 1
        return tid

    @staticmethod
    def tid_list_1day24(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S") #time.struct_time(tm_year=2010, tm_mon=6, tm_mday=5, tm_hour=0, tm_min=11, tm_sec=12, tm_wday=5, tm_yday=156, tm_isdst=-1)
        tid= tm.tm_hour
        return tid

    @staticmethod
    def tid_list_48_gowalla(tmd):
        tm = time.strptime(tmd, "%H:%M") #time.struct_time(tm_year=2010, tm_mon=6, tm_mday=5, tm_hour=0, tm_min=11, tm_sec=12, tm_wday=5, tm_yday=156, tm_isdst=-1)
        if tm.tm_min < 30:
            tid = tm.tm_hour * 2
        else:
            tid = tm.tm_hour * 2 + 1
        return tid

    def prepare_neural_data(self):
        # output_data=[]
        # for u in self.uid_list:
        #     sessions = self.data_filter[u]['sessions']
        #     for sid in sessions:
        #         a=[[self.uid_list[u][0], self.vid_list[p[0]][0], p[1], self.vid_lookup[self.vid_list[p[0]][0]][0],
        #          self.vid_lookup[self.vid_list[p[0]][0]][1]] for p in sessions[sid]]
        #         output_data.extend(a)
        #
        # save_path = '../STRNNdata/Toyota.txt'
        # ff = pd.DataFrame(output_data)
        # ff.to_csv(save_path, header=None, index=False)

        # hotel: 4bf58dd8d48988d1fa931735 [125, 742]
        # subway: 4bf58dd8d48988d1fd931735
        # bus station: 4bf58dd8d48988d1fe931735 [18, 4319]
        # university: 4bf58dd8d48988d1ae941735 [136, 2394]
        # university: 4bf58dd8d48988d1ae941735 [136, 2394]
        # Food & Drink Shop: 4bf58dd8d48988d118951735 [8, 5843]
        university_dict={}
        bustation_dict={}
        for i in range(24):
            university_dict[i]=[]
            bustation_dict[i] = []

        for u in self.uid_list:
            sessions = self.data_filter[u]['sessions']
            sessions_tran = {}
            sessions_id = []


            session_id=[]
            split_id = int(np.floor(self.train_split * len(sessions)))
            train_id = session_id[:split_id]
            test_id = session_id[split_id:]
            temptrain = []
            temptest = []
            for sid in sessions:
                if sid < split_id :
                    temptrain.append([sessions[sid][0],sid])
                else:
                    temptest.append(sessions[sid][0])
            for sid in sessions:
                # sessions_tran[sid] = [
                #     [self.vid_list[p[0]][0], self.tid_list_48(p[1]), self.vid_lookup[self.vid_list[p[0]][0]][0],
                #      self.vid_lookup[self.vid_list[p[0]][0]][1], self.vid_list[p[0]][1], self.catid_list[p[-1]][0]] for p in sessions[sid]]#p[0]是POI ID, p[1]是time, self.vid_list[p[0]][0]给出在列表中的位置
                sessions_tran[sid] = [
                    [self.vid_list[p[0]][0], self.tid_list_1day48(p[1]), self.vid_lookup[self.vid_list[p[0]][0]][0],
                     self.vid_lookup[self.vid_list[p[0]][0]][1], self.vid_list[p[0]][1], self.vid_list[p[0]][-2]] for p in sessions[sid]]#p[0]是POI ID, p[1]是time, self.vid_list[p[0]][0]给出在列表中的位置
                sessions_id.append(sid)
                for p in sessions[sid]:
                    if self.vid_list[p[0]][-2]==8:
                        university_dict[self.tid_list_1day24(p[1])].append(0)
                    if self.vid_list[p[0]][-2]==18:
                        bustation_dict[self.tid_list_1day24(p[1])].append(0)

            split_id = int(np.floor(self.train_split * len(sessions_id))) #train_split=0.8, np.floor向下取整
            train_id = sessions_id[:split_id]
            test_id = sessions_id[split_id:]
            pred_len = sum([len(sessions_tran[i]) - 1 for i in train_id]) #训练集拥有的label数量
            valid_len = sum([len(sessions_tran[i]) - 1 for i in test_id]) #测试集拥有的label数量
            train_loc = {}
            for i in train_id: #train_id就是session_id的前百分之八十，每个session都是从0开始的
                for sess in sessions_tran[i]:
                    if sess[0] in train_loc:
                        train_loc[sess[0]] += 1
                    else:
                        train_loc[sess[0]] = 1
            #用train_loc记录所有train_id(前百分之八十的会话)包含多少种不同的poi以及他们的出现次数
            # calculate entropy
            entropy = entropy_spatial(sessions) #这他妈的等于把上面的循环又执行了一遍，只不过针对的是整个session,就是一个求和对所有加权平均取log

            # calculate location ratio
            train_location = []
            for i in train_id:
                train_location.extend([s[0] for s in sessions[i]])
            train_location_set = set(train_location) #在我看来就是把所有签到的POI汇总然后去重
            test_location = []
            for i in test_id:
                test_location.extend([s[0] for s in sessions[i]])
            test_location_set = set(test_location)
            whole_location = train_location_set | test_location_set #set的合并方式
            test_unique = whole_location - train_location_set
            location_ratio = len(test_unique) / len(whole_location)

            # calculate radius of gyration
            lon_lat = []
            for pid in train_location:
                try:
                    lon_lat.append(self.pid_loc_lat[pid]) #从大列表中找到 POI对应的经纬度
                except:
                    print(pid)
                    print('error')
            lon_lat = np.array(lon_lat)
            center = np.mean(lon_lat, axis=0, keepdims=True)
            center = np.repeat(center, axis=0, repeats=len(lon_lat))

            # rg = np.sqrt(np.mean(np.sum((lon_lat - center) ** 2, axis=1, keepdims=True), axis=0))[0] #貌似是算每个POI到中心点的距离
            rg=0
            session_train=[]
            for j in range(train_id[-1] + 1):
                session_train.extend(sessions_tran[j])
            self.data_neural[self.uid_list[u][0]] = {'sessions': sessions_tran, 'sessions_train': session_train, 'train': train_id, 'test': test_id,
                                                     'pred_len': pred_len, 'valid_len': valid_len,
                                                     'train_loc': train_loc, 'explore': location_ratio,
                                                     'entropy': entropy, 'rg': rg}
        bustation_list=[]
        university_list=[]
        for i in bustation_dict.keys():
            bustation_list.append(len(bustation_dict[i]))
            university_list.append(len(university_dict[i]))
        jj = pd.DataFrame(university_list)
        # jj.to_csv("foodanddrink_cattimefreq_list_tky.txt", index=False)
        jjj = pd.DataFrame(bustation_list)
        # jjj.to_csv("busstation_cattimefreq_list_tky.txt", index=False)

    def build_users_locations_dict_train(self):
        # self.user_filter3=self.user_filter3[0:100]
        num_user=len(self.uid_list)
        user_id=[self.uid_list[x][0] for x in self.uid_list]
        user_graph = pd.DataFrame(np.zeros(num_user ** 2).reshape(num_user, num_user),
                                  index=set(user_id),
                                  columns=set(user_id))

        
        for u in tqdm(self.uid_list):
            sessions = self.data_neural[self.uid_list[u][0]]['sessions_train']
            source_poi= set([x[0] for x in sessions])
            for u_d in self.uid_list:
                if user_graph.loc[self.uid_list[u][0], self.uid_list[u_d][0]] > 0 or user_graph.loc[
                    self.uid_list[u_d][0], self.uid_list[u][0]] > 0:
                    continue
                if u==u_d:
                    continue
                sessions_d = self.data_neural[self.uid_list[u_d][0]]['sessions_train']
                target_poi = set([x[0] for x in sessions_d])
                intersection = source_poi & target_poi
                union = source_poi | target_poi
                duplicates = len(intersection)
                total = len(union)
                score=duplicates/total
                if score>=0.2:
                    user_graph.loc[self.uid_list[u][0], self.uid_list[u_d][0]] = 1
                    user_graph.loc[self.uid_list[u_d][0], self.uid_list[u][0]] = 1
                    
        user_graph.to_csv("../dataset/foursquaregraph/raw/Graph_user.csv", index=False)
        # ---- Graph_user.csv generation                                  
        print("Graph_user.csv saved!")

        for u in self.uid_list:
            sessions = self.data_neural[self.uid_list[u][0]]['sessions']
            train_id = self.data_neural[self.uid_list[u][0]]['train'][-1]+1
            # sessions = {x: sessions[x] for x in range(int(len(sessions)*0.1))}
            for sid in sessions:
                if sid == train_id:
                    break
                # 对于定长的graphormer
                poi = [p[0] for p in sessions[sid]]  # 提取所有POI
                lat = [p[2] for p in sessions[sid]]
                lon = [p[3] for p in sessions[sid]]
                cat = [p[-1] for p in sessions[sid]]
                index = 0
                for p in poi:
                    if p not in self.vid_list_train:  # self.vid_list = {'unk': [0, -1]} 又是统计POI出现的次数并给他一个编号，而且编号共同，所有用户同享
                        self.vid_list_lookup_train[len(self.vid_list_train)] = p
                        self.vid_list_train[p] = [len(self.vid_list_train), 1, lat[index], lon[index], cat[index], 0]
                        temp = len(self.vid_list_train[p]) - 2
                    else:
                        self.vid_list_train[p][1] += 1
                    index += 1
                for c in cat:
                    if c not in self.catid_list_train:  # self.vid_list = {'unk': [0, -1]} 又是统计POI出现的次数并给他一个编号，而且编号共同，所有用户同享
                        self.catid_list_lookup_train[len(self.catid_list_train)] = c
                        self.catid_list_train[c] = [len(self.catid_list_train), 1]
                    else:
                        self.catid_list_train[c][1] += 1
        for i in self.vid_list_train:
            if i == 'unk':
                continue
            self.vid_list_train[i][-1] = self.catid_list_train[self.vid_list_train[i][temp]][1]
            self.vid_list_train[i][temp] = self.catid_list_train[self.vid_list_train[i][temp]][0]

    def save_txt(str_list: list, name):
        with open(name, 'w', encoding='utf-8') as f:
            for i in str_list:
                f.write(i + '\n')


    def prepare_global_data(self):
        poi_list=[]
        poi_name=[]
        cat_name = []
        poi_list_train = []
        catid_list_train={}
        # for i in self.catid_list:
        #     if i=='unk':
        #         continue
        #     cat_name.append(self.catid_list[i][0])
        for i in self.vid_list:
            if i=='unk':
                continue
            # if poi_list==[]:
            #     if self.vid_list[i][0] not in self.vid_list_train:
            #         cat_freq=self.catid_list_train[self.vid_list[i][-2]][1]
            #         poi_list=[self.vid_list[i]]
            #         poi_list[-1][-1]=cat_freq
            #         poi_list[-1][1]=0
            #     else:
            #         poi_list = [self.vid_list_train[self.vid_list[i][0]]]
            # else:
            #     if self.vid_list[i][0] not in self.vid_list_train:
            #         cat_freq = self.catid_list_train[self.vid_list[i][-2]][1]
            #         poi_list.append(self.vid_list[i])
            #         poi_list[-1][-1] = cat_freq
            #         poi_list[-1][1] = 0
            #     else:
            #         poi_list.append(self.vid_list_train[self.vid_list[i][0]])
            poi_list.append(self.vid_list[i])
            poi_name.append(self.vid_list[i][0])
            
        for i in self.vid_list:
            if i == 'unk':
                continue
            c=self.vid_list[i][-2]
            if c not in catid_list_train:  # self.vid_list = {'unk': [0, -1]} 又是统计POI出现的次数并给他一个编号，而且编号共同，所有用户同享
                catid_list_train[c] = [len(catid_list_train)+1, 1]
            else:
                catid_list_train[c][1] += 1
        for i in range(len(poi_list)):
            poi_list[i][-2]=catid_list_train[poi_list[i][-2]][0]

        freq_list=[]
        for i in self.vid_list.keys():
            if i =='unk':
                continue
            else:
                freq_list.append(self.vid_list[i][1])
        time_list=[]
        for i in self.data_neural.keys():
            for j in self.data_neural[i]['sessions'].keys():
                for k in range(len(self.data_neural[i]['sessions'][j])):
                    if self.data_neural[i]['sessions'][j][k][1]>23:
                        time_list.append(self.data_neural[i]['sessions'][j][k][1]-24)
                    else:
                        time_list.append(self.data_neural[i]['sessions'][j][k][1])

        jj = pd.DataFrame(time_list)
        # jj.to_csv("time_list_gowalla.txt", index=False)
        jjj = pd.DataFrame(freq_list)
        # jjj.to_csv("freq_list_gowalla.txt", index=False)

        #-----------------------------------------
        # loc_user=0
        # for i in self.data_neural.keys():
        #     loc_user+=len(self.data_neural[i]['train_loc'])
        # loc_users=loc_user/len(self.data_neural)
        #
        # session_count=0
        # check_in_POI=0
        # for i in self.data_filter.keys():
        #     session_count+=self.data_filter[i]['sessions_count']
        #     for j in self.data_filter[i]['sessions']:
        #         check_in_POI+=len(self.data_filter[i]['sessions'][j])
        #--------------------------------------------------------------------

        for i in catid_list_train:
            if i=='unk':
                continue
            cat_name.append(catid_list_train[i][0])

        pd_poi=pd.DataFrame(poi_list)
        head=['POI ID', 'checkin_cnt', 'lat', 'lon', 'cat', 'cat_freq']
        num_cat = len(catid_list_train)
        cat_graph = pd.DataFrame(np.zeros(num_cat ** 2).reshape(num_cat, num_cat),
                                 index=set(cat_name),
                                 columns=set(cat_name))
        
        for i in tqdm(self.data_neural):
            for j in self.data_neural[i]['sessions']:
                if j == self.data_neural[i]['train'][-1]+1:
                    break
                for k in range(len(self.data_neural[i]['sessions'][j])):
                    if k>0:
                        # if poi_graph.loc[loc_old, loc_new] == 0:
                        # if self.data_neural[i]['sessions'][j][k][-1]>286:
                        #     print(1)
                        loc_new = self.data_neural[i]['sessions'][j][k][-1]
                        loc_old = self.data_neural[i]['sessions'][j][k-1][-1]
                        try:
                            cat_graph.loc[loc_old, loc_new] += 1
                        except:
                            print("Exception!")

        cat_graph.to_csv("../dataset/foursquaregraph/raw/Graph_cat.csv", index=False)
        # ---- Graph_cat.csv generation
        print("Graph_cat.csv saved!")

        num_poi=len(self.vid_list)-1
        poi_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                 index=set(poi_name),
                                 columns=set(poi_name))
        for i in tqdm(self.data_neural):
            for j in self.data_neural[i]['sessions']:
                if j == self.data_neural[i]['train'][-1]+1:
                    break
                for k in range(len(self.data_neural[i]['sessions'][j])):
                    if k>0:
                        # if poi_graph.loc[loc_old, loc_new] == 0:
                        loc_new = self.data_neural[i]['sessions'][j][k][0]
                        loc_old = self.data_neural[i]['sessions'][j][k-1][0]
                        poi_graph.loc[loc_old, loc_new] += 1

        poi_id=[x[0] for x in poi_list]
        dist_graph = pd.DataFrame(np.zeros(num_poi ** 2).reshape(num_poi, num_poi),
                                  index=set(poi_id),
                                  columns=set(poi_id))

        # dist_graph option 1
        for i in tqdm(poi_list):
            latlon1=i[2:4]
            for j in poi_list:
                latlon2 = j[2:4]
                dist=LLs2Dist(float(latlon1[0]),float(latlon1[1]),float(latlon2[0]),float(latlon2[1]))
                if dist<=3 and dist>0:
                    dist_graph.loc[i[0], j[0]]=1
                    dist_graph.loc[j[0], i[0]]= 1

        # dist_graph option 2
        # for i in tqdm(poi_list):
        #     latlon1 = i[2:4]
        #     for j in poi_list:
        #         latlon2 = j[2:4]
        #         dist = LLs2Dist(float(latlon1[0]), float(latlon1[1]), float(latlon2[0]), float(latlon2[1]))
        #         # dist = geodistance(float(latlon1[1]), float(latlon1[0]), float(latlon2[1]), float(latlon2[0]))
        #         if dist < 1:
        #             dist = 1
        #         dist_graph.loc[i[0], j[0]] = dist
        #         dist_graph.loc[j[0], i[0]] = dist
        
        # dist_graph option 3
        # for i in tqdm(poi_list):
        #     latlon1=i[2:4]
        #     dist_list = []
        #     dist_index= []
        #     for j in poi_list:
        #         if i[0]==j[0]:
        #             continue
        #         latlon2 = j[2:4]
        #         dist=LLs2Dist(float(latlon1[0]),float(latlon1[1]),float(latlon2[0]),float(latlon2[1]))
        #         if len(dist_list)<5:
        #             dist_index.append(j[0])
        #             dist_list.append(dist)
        #         else:
        #             temp=np.array(dist_list)
        #             if dist<max(dist_list):
        #                 dist_index[temp.argmax()]=j[0]
        #                 dist_list[temp.argmax()] = dist

        #         if dist<=3 and dist>0:
        #             dist_graph.loc[i[0], j[0]]=1
        #             dist_graph.loc[j[0], i[0]]= 1
                
        #     for j in range(len(dist_index)):
        #         dist_graph.loc[i[0], dist_index[j]] = 1
        #         dist_graph.loc[dist_index[j], i[0]]= 1


        dist_graph.to_csv("../dataset/foursquaregraph/raw/Graph_dist.csv", index=False)
        # ---- Graph_dist.csv generation
        print("Graph_dist.csv saved!")

        pd_poi.to_csv("../dataset/foursquaregraph/raw/Graph_poi.csv", header=head, index=False)
        # ---- Graph_poi.csv generation
        print("Graph_poi.csv saved!")
        
        poi_graph.to_csv("../dataset/foursquaregraph/raw/Graph_adj.csv", index=False)
        # ---- Graph_adj.csv generation
        print("Graph_adj.csv saved!")


    # ############# 6. save variables
    def get_parameters(self):
        parameters = {}
        parameters['TWITTER_PATH'] = self.TWITTER_PATH
        parameters['SAVE_PATH'] = self.SAVE_PATH

        parameters['trace_len_min'] = self.trace_len_min
        parameters['location_global_visit_min'] = self.location_global_visit_min
        parameters['hour_gap'] = self.hour_gap
        parameters['min_gap'] = self.min_gap
        parameters['session_max'] = self.session_max
        parameters['filter_short_session'] = self.filter_short_session
        parameters['sessions_min'] = self.sessions_count_min
        parameters['train_split'] = self.train_split

        return parameters

    def save_variables(self):
        foursquare_dataset = {'data_neural': self.data_neural, 'vid_list': self.vid_list, 'uid_list': self.uid_list,
                              'parameters': self.get_parameters(), 'data_filter': self.data_filter,
                              'vid_lookup': self.vid_lookup}
        pickle.dump(foursquare_dataset, open(self.SAVE_PATH + self.save_name + '.pk', 'wb'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_min', type=int, default=10, help="raw trace length filter threshold") #10
    parser.add_argument('--global_visit', type=int, default=10, help="location global visit threshold") #10
    parser.add_argument('--hour_gap', type=int, default=24, help="maximum interval of two trajectory points") #72
    parser.add_argument('--min_gap', type=int, default=0, help="minimum interval of two trajectory points") #10
    parser.add_argument('--session_max', type=int, default=10000, help="control the length of session not too long")
    parser.add_argument('--session_min', type=int, default=3, help="control the length of session not too short") #5
    parser.add_argument('--sessions_min', type=int, default=2, help="the minimum amount of the good user's sessions") #5
    parser.add_argument('--train_split', type=float, default=0.8, help="train/test ratio")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_generator = DataFoursquare(trace_min=args.trace_min, global_visit=args.global_visit,
                                    hour_gap=args.hour_gap, min_gap=args.min_gap,
                                    session_min=args.session_min, session_max=args.session_max,
                                    sessions_min=args.sessions_min, train_split=args.train_split)
    parameters = data_generator.get_parameters()
    print('############PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
    print('############START PROCESSING:')
    
    print('1) load trajectory from {}'.format(data_generator.TWITTER_PATH))
    data_generator.load_trajectory_from_tweets()
    
    print('2) filter users')
    data_generator.filter_users_by_length()
    
    print('3) build users/locations dictionary')
    data_generator.build_users_locations_dict()
    
    print("4) Load venues and venues lookup")
    data_generator.load_venues()
    data_generator.venues_lookup()

    print('5) prepare data for neural network')
    data_generator.prepare_neural_data()
    
    print('6) build training users/locations dictionary')
    data_generator.build_users_locations_dict_train()
    
    print('7) prepare data for global network')
    data_generator.prepare_global_data()
    
    print('8) save prepared data')
    data_generator.save_variables()
    
    print("---- General report ----")
    print('raw users:{} raw locations:{}'.format(
        len(data_generator.data), len(data_generator.venues)))
    print('final users:{} final locations:{}'.format(
        len(data_generator.data_neural), len(data_generator.vid_list)))
