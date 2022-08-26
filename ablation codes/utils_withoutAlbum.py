import numpy as np
from collections import defaultdict
import os
import pickle
import copy
import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import datetime
import time
import argparse
import pickle
import re
import random
import sys

def data_input(opt):
    # 返回测试集和验证集数据

    BASE_DIR = './datasets'
    #BASE_DIR = '/user-data/MusicSeq'
    dataset_name=opt.dataset
    if dataset_name == 'last':
        freq = 50
        topu = 900
        PATH = 'lastfm'
        n_music = 66407
    elif dataset_name == 'xiami':
        freq = 10
        topu = 4000
        PATH = 'xiami'
        n_music = 64334
    elif dataset_name == '30Music':
        freq = 50
        topu = 3000
        PATH = '30Music'
        n_music = 90868

    if opt.valid_portion == 0:
        test_user_item_record = os.path.join(BASE_DIR, PATH, 'baseline_te_new_freq{}.lst'.format(freq))
        train_user_item_record = os.path.join(BASE_DIR, PATH, 'baseline_tr_freq{}.lst'.format(freq))
    else:
        test_user_item_record = os.path.join(BASE_DIR, PATH,
                                             'baseline_te_new_freq{}_partition{}.lst'.format(freq, opt.valid_portion))
        train_user_item_record = os.path.join(BASE_DIR, PATH,
                                              'baseline_tr_freq{}_partition{}.lst'.format(freq, opt.valid_portion))

    music2artist_dic_record = os.path.join(BASE_DIR, PATH,
                                           '{}_music_index2artist_freq{}_topu{}'.format(dataset_name, freq, topu))
    music2album_dic_record = os.path.join(BASE_DIR, PATH,
                                          '{}_music_index2album_freq{}_topu{}'.format(dataset_name, freq, topu))
    music2artist_dic = pickle.load(open(music2artist_dic_record, 'rb'))
    music2album_dic = pickle.load(open(music2album_dic_record, 'rb'))

    test_user_item_file = open(test_user_item_record, 'rb')
    train_user_item_file = open(train_user_item_record, 'rb')

    test_lines = test_user_item_file.readlines()
    train_lines = train_user_item_file.readlines()

    # 读入时,对于item的下标+1,0号下标用作item,usr不受影响
    train_user, train_music, train_artist, train_album = [], [], [], []
    test_user, test_music, test_artist, test_album = [], [], [], []

    # 存储所有train数据中，用户听过的歌曲
    usr2music_record_dic = defaultdict(set)

    flag = 1
    train_lenth,test_lenth=0,0
    user_set=set()
    train_missed_artist, train_missed_album=0,0
    train_missed_artist_set, train_missed_album_set=set(),set()
    for line in train_lines:
        if flag:
            flag = 0
            continue
        line = line.decode()
        user_id = int(line.split(',')[0])
        user_set.add(user_id)

        items_id = line.split(',')[1].split(':')
        int_items_id = list(map(int, items_id))  # 批量转换为int类型
        int_items_id_plus = np.array(int_items_id) + 1  # 下标全部+1
        train_lenth+=len(int_items_id)
        for item in int_items_id_plus:  # 加入usr-item字典
            usr2music_record_dic[user_id].add(item)

        train_user.append(user_id)
        train_music.append(int_items_id_plus.tolist())

        for x in int_items_id:
            if music2artist_dic[x]==-1:
                train_missed_artist+=1
                train_missed_artist_set.add(x)
            if music2album_dic[x]==-1:
                train_missed_album+=1
                train_missed_album_set.add(x)

        train_artist.append([int(music2artist_dic[x]) + 1 for x in int_items_id])  # 字典内容为原始id的映射
        train_album.append([int(music2album_dic[x]) + 1 for x in int_items_id])
        #break

    test_missed_artist, test_missed_album = 0, 0
    test_missed_artist_set, test_missed_album_set = set(), set()
    for line in test_lines:
        line = line.decode()
        user_id = int(line.split(',')[0])
        user_set.add(user_id)

        items_id = line.split(',')[1].split(':')

        int_items_id = list(map(int, items_id))  # 批量转换为int类型
        int_items_id_plus = np.array(int_items_id) + 1  # 下标全部+1
        test_lenth+=len(int_items_id)
        test_user.append(user_id)
        test_music.append(int_items_id_plus.tolist())

        for x in int_items_id:
            if music2artist_dic[x]==-1:
                test_missed_artist+=1
                test_missed_artist_set.add(x)
            if music2album_dic[x]==-1:
                test_missed_album+=1
                test_missed_album_set.add(x)

        test_artist.append([int(music2artist_dic[x]) + 1 for x in int_items_id])
        test_album.append([int(music2album_dic[x]) + 1 for x in int_items_id])
        #break

    test_user_item_file.close()
    train_user_item_file.close()


    train_data = train_user, train_music, train_artist, train_album
    test_data = test_user, test_music, test_artist, test_album

    train_data = random_sampling(train_data, rate=opt.drop_portion)
    train_data = data_split(train_data, usr2music_record_dic,opt, '', is_test=False)

    test_new_data=copy.deepcopy(test_data)
    test_new_data = data_split(test_new_data, usr2music_record_dic,opt,'next-new-item', is_test=True)
    test_data = data_split(test_data, usr2music_record_dic,opt,'next-one-item', is_test=True)

    return train_data, test_data, test_new_data


def data_split(data, usr2music_record_dic, opt,mode, is_test=False):
    # 对于pickle读入的序列化数据
    # 0：用户id 1：music 2：artist
    #print("---------------data_spliting---------------")
    windowLenth = opt.windowLenth
    data_size = opt.data_size
    step = opt.slide_step  # 只影响train

    user, target, music_seq, artist_seq, album_seq = [], [], [], [], []

    if is_test == False:
        # 数据集为train
        for user_id, music_slice, artist_slice, album_slice in zip(data[0], data[1], data[2], data[3]):
            for i in range(windowLenth, int(len(music_slice) * data_size), step):
                user.append(user_id)
                target += [music_slice[i]]
                music_seq.append(music_slice[i - windowLenth:i])
                artist_seq.append(artist_slice[i - windowLenth:i])
                album_seq.append(album_slice[i - windowLenth:i])
    else:
        # 数据集为test
        if mode == 'next-new-item':
            # 窗口内和训练集中都未出现
            for user_id, music_slice, artist_slice, album_slice in zip(data[0], data[1], data[2], data[3]):
                for i in range(windowLenth, int(len(music_slice))):
                    t = music_slice[i]
                    if t in usr2music_record_dic[user_id] or t in music_slice[i - windowLenth:i]:
                        continue
                    user.append(user_id)
                    target += [music_slice[i]]
                    music_seq.append(music_slice[i - windowLenth:i])
                    artist_seq.append(artist_slice[i - windowLenth:i])
                    album_seq.append(album_slice[i - windowLenth:i])

        elif mode == 'next-recent-new-item':
            # 只有滑动窗口内不出现
            for user_id, music_slice, artist_slice, album_slice in zip(data[0], data[1], data[2], data[3]):
                for i in range(windowLenth, int(len(music_slice))):
                    t = music_slice[i]
                    if t in music_slice[i - windowLenth:i]:
                        continue
                    user.append(user_id)
                    target += [music_slice[i]]
                    music_seq.append(music_slice[i - windowLenth:i])
                    artist_seq.append(artist_slice[i - windowLenth:i])
                    album_seq.append(album_slice[i - windowLenth:i])

        elif mode == 'next-one-item':
            for user_id, music_slice, artist_slice, album_slice in zip(data[0], data[1], data[2], data[3]):
                for i in range(windowLenth, int(len(music_slice))):
                    user.append(user_id)
                    target += [music_slice[i]]
                    music_seq.append(music_slice[i - windowLenth:i])
                    artist_seq.append(artist_slice[i - windowLenth:i])
                    album_seq.append(album_slice[i - windowLenth:i])

    return user, target, music_seq, artist_seq, album_seq


def random_sampling(data, rate=0):
    # 对于划分前的序列进行随机采样，求子序列，类似于drop
    # 算法流程：创建下标列表，随机采样后，转化为set，遍历每个元素，根据下标是否在set中加入序列
    # data格式：user-music-artist-album
    if rate == 0:
        return data

    random.seed(42)

    # 根据指定rate进行切分
    new_usr, new_music, new_artist, new_album = [], [], [], []

    for usr, music, artist, album in zip(data[0], data[1], data[2], data[3]):
        total_num = len(music)
        index = list(range(total_num))
        new_index = set(random.sample(index, int(total_num * (1-rate))))

        new_music_slice = [music[idx] for idx in new_index]
        new_artist_slice = [artist[idx] for idx in new_index]
        new_album_slice = [album[idx] for idx in new_index]

        new_usr.append(usr)
        new_music.append(new_music_slice)
        new_artist.append(new_artist_slice)
        new_album.append(new_album_slice)

    return new_usr, new_music, new_artist, new_album


def build_graph(music_seq, artist_seq):
    matrix=[]
    music_alias_input, artist_alias_input = [], []
    artist_mask_input= []
    # 0号节点用于padding，将所有item标号+1；根据输入的音乐序列，创建对应的alias_input序列，对应item中的下标用于从hidden获取向量进行顺序排布
    items, len_list = [], []
    # 生成一张图中对应的item,对应用于直接获取embedding的item的表示

    for music_slice, artist_slice in zip(music_seq, artist_seq):
        a = np.unique(music_slice + artist_slice)  # album中包含0表示不存在
        items.append(a)
        len_list.append(a.shape[0] + 1)  # +1为用户节点的占位

    # 根据每张图中的节点个数：1.获取所有图的节点个数 2.获取最多节点个数 3.得到mask列表
    max_node = max(len_list)  # 相对于items的长度+1用于创建邻接矩阵

    # item padding
    for i in range(len(items)):
        items[i] = np.pad(items[i], (0, max_node - len_list[i]), 'constant', constant_values=(0))

    for i in range(len(items)):
        # 根据music_seq获取item中的下标
        # 对应hidden中下标，hidden中包含usr，music，album，artist
        music_alias_input.append([np.where(items[i] == m)[0][0] + 1 for m in music_seq[i]])
        artist_alias_input.append([np.where(items[i] == m)[0][0] + 1 for m in artist_seq[i]])
        mask = []
        for m in artist_seq[i]:
            if m != 0:
                mask += [1]
            else:
                mask += [0]
        artist_mask_input.append(mask)

        # 邻接矩阵下标0：usr; 下标1~max_node：item(music,artist,album)
        u_A = np.zeros((max_node, max_node))
        # usr → music[0],artist[0],album[0]
        u_A[0][np.where(items[i] == music_seq[i][0])[0][0] + 1] = 1
        u_A[0][np.where(items[i] == artist_seq[i][0])[0][0] + 1] = 1

        pre = -1  # 存储之前的那首歌曲
        for music, artist in zip(music_seq[i], artist_seq[i]):
            if pre!= -1:
                pre_music_pos=np.where(items[i] == pre)[0][0] + 1
            now_music_pos=np.where(items[i] == music)[0][0] + 1
            artist_pos=np.where(items[i] == artist)[0][0] + 1

            # pre_music → music
            if pre != -1:
                u_A[pre_music_pos][now_music_pos]=1
            pre = music

            # artist → music
            if artist !=0 :
                u_A[artist_pos][now_music_pos]=1

        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1

        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_out, u_A_in]).transpose()
        matrix.append(u_A)

    return matrix, items, music_alias_input, artist_alias_input,artist_mask_input


class Data():
    def __init__(self, data, opt, shuffle=False):

        usr = data[0]
        target = data[1]
        music_seq = data[2]
        artist_seq = data[3]
        album_seq = data[4]
        matrix, items, music_alias_input, artist_alias_input,artist_mask_input = build_graph(music_seq, artist_seq)

        self.usr = np.asarray(usr)
        self.target = np.asarray(target)
        self.items = np.asarray(items)
        self.music_alias_input = np.asarray(music_alias_input)
        self.artist_alias_input = np.asarray(artist_alias_input)
        self.artist_mask_input = np.asarray(artist_mask_input)
        self.matrix = np.asarray(matrix)

        self.windowLenth = opt.windowLenth
        self.shuffle = shuffle

    def generate_batch(self, batch_size):  # 训练数据输入多少批次
        length = len(self.target)
        if self.shuffle:
            shuffled_arg = np.arange(length)
            np.random.shuffle(shuffled_arg)
            self.usr = self.usr[shuffled_arg]
            self.target = self.target[shuffled_arg]
            self.items = self.items[shuffled_arg]
            self.matrix = self.matrix[shuffled_arg]

            self.music_alias_input = self.music_alias_input[shuffled_arg]
            self.artist_alias_input = self.artist_alias_input[shuffled_arg]
            self.artist_mask_input = self.artist_mask_input[shuffled_arg]

        n_batch = int(length / batch_size)
        if length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(length - batch_size * (n_batch - 1))]
        return slices