# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import pickle
import os
import json
import numpy as np
import copy

# 导入上级目录下的模块 baseline_config
import sys

import baseline_config


def get_item_lengths_list(data):
    item_length_list_dict = dict()
    user_group = data.groupby(['user_id'])
    # short sequence comes first
    for user_id, length in user_group.size().sort_values().iteritems():
        # oldest data comes first
        temp_user_data = user_group.get_group(user_id)
        temp_time_seq = copy.deepcopy(pd.to_datetime(temp_user_data['timestamp']))
        temp_user_data.loc[:, 'timestamp_new'] = temp_time_seq
        user_data = temp_user_data.sort_values(by='timestamp_new')
        # user_data = user_data[user['tranname'].notnull()]
        music_seq = user_data['tran_name_id']
        time_seq = user_data['timestamp_new']
        # filter the null data.
        # time_seq = time_seq[time_seq.notnull()]
        # 应该根据相同的标准进行过滤
        time_seq = time_seq[music_seq.notnull()]
        music_seq = music_seq[music_seq.notnull()]
        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = time_seq.diff(-1).astype('timedelta64[s]') * -1
        item_seq = music_seq.tolist()

        delta_time = delta_time.tolist()
        delta_time[-1] = 0  # 原本的delta_time[-1] 是 nan
        for item, time in zip(item_seq, delta_time):
            time = int(time)
            if time <= 0 or time >= 3600:
                continue
            if item in item_length_list_dict:
                length_list_temp = item_length_list_dict.get(item)
            else:
                length_list_temp = []
            length_list_temp.append(time)
            item_length_list_dict[item] = length_list_temp
    return item_length_list_dict


def get_item_length(item_length_list_dict, item2index):
    item_length = dict()
    # short sequence comes first
    for item in item_length_list_dict.keys():
        if item in item2index:
            length_list = item_length_list_dict[item]
            max_length = max(length_list, key=length_list.count)
            if length_list.count(max_length) >= 2 and max_length < 3600:
                item_length[item] = max_length

            else:
                print(item + ": ")
                print(length_list)
                # item_length[item] = -1
    return item_length


def generate_data(freq_n_item, top_n_user, min_length, max_length, data, BASE_DIR, DATA_SOURCE,partition, small_data=False):
    # NORM_METHOD = 'origin'

    # NORM_METHOD = 'log'
    # (x-min)/(max-min)
    # NORM_METHOD = 'mm'
    # /3600
    NORM_METHOD = 'hour'

    # filter bad lines in original file
    # path_filtered = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname-filtered.tsv')
    # path_filtered = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname-small.tsv')
    # path_filtered = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname.tsv')
    name_suffix = 'freq{}-min{}-max{}-{}-{}.lst'.format(freq_n_item, min_length,max_length,NORM_METHOD,int(partition*10))
    if small_data:
        tr_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'small_tr-user-item-time-' + name_suffix)
        # next item
        te_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'small_te-user-old-item-time-' + name_suffix)
        # next new item
        te_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'small_te-user-new-item-time-' + name_suffix)
    else:
        tr_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'tr-user-item-time-' + name_suffix)
        # next item
        te_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te-user-old-item-time-' + name_suffix)
        # next new item
        te_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te-user-new-item-time-' + name_suffix)

    # normal baseline
    tr_baseline_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                     'baseline_tr_freq{}_partition{}.lst'.format(freq_n_item,
                                                                                                 int(partition * 10)))
    te_baseline_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                         'baseline_te_old_freq{}_partition{}.lst'.format(freq_n_item,
                                                                                                         int(partition * 10)))
    te_baseline_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                         'baseline_te_new_freq{}_partition{}.lst'.format(freq_n_item,
                                                                                                         int(partition * 10)))

    # lstm baseline
    tr_lstm_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'lstm_tr_freq{}_partition{}.lst'.format(freq_n_item,int(partition*10)))
    te_lstm_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'lstm_te_old_freq{}_partition{}.lst'.format(freq_n_item,int(partition*10)))
    te_lstm_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'lstm_te_new_freq{}_partition{}.lst'.format(freq_n_item,int(partition*10)))

    index2item_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   'last_music_index2item_freq' + str(freq_n_item) + '_topu' + str(top_n_user))
    item2index_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   'last_music_item2index_freq' + str(freq_n_item) + '_topu' + str(top_n_user))
    index2length_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                     'last_music_index2length_freq' + str(freq_n_item) + '_topu' + str(top_n_user))
    index2words_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                    'last_music_index2words_freq' + str(freq_n_item) + '_topu' + str(top_n_user))

    item_length_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_item_length_freq' + str(freq_n_item))
    item_length_list_dict_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_item_length_list_dict')

    out_tr_uit = open(tr_user_item_time_record, 'w', encoding='utf-8')
    out_te_old_uit = open(te_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_new_uit = open(te_user_new_item_time_record, 'w', encoding='utf-8')

    # baseline
    out_tr_baseline_uit = open(tr_baseline_user_item_time_record, 'w', encoding='utf-8')
    #首行
    out_tr_baseline_uit.write(str(top_n_user) + ", " + str(baseline_config.top_n_item_dict_lastfm[freq_n_item]) + '\n')

    out_te_baseline_old_uit = open(te_baseline_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_baseline_new_uit = open(te_baseline_user_new_item_time_record, 'w', encoding='utf-8')

    # lstm baseline
    out_tr_lstm_uit = open(tr_lstm_user_item_time_record, 'w', encoding='utf-8')
    #首行
    out_tr_lstm_uit.write(str(top_n_user) + ", " + str(baseline_config.top_n_item_dict_lastfm[freq_n_item]) + '\n')

    out_te_lstm_old_uit = open(te_lstm_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_lstm_new_uit = open(te_lstm_user_new_item_time_record, 'w', encoding='utf-8')

    if os.path.exists(index2item_path) and os.path.exists(item2index_path):
        index2item = pickle.load(open(index2item_path, 'rb'))
        item2index = pickle.load(open(item2index_path, 'rb'))
        print('Total music and user %d' % len(index2item))
    else:
        print('Build users index')
        sorted_user_series = data.groupby(['user_id']).size().sort_values(ascending=False)
        print('sorted_user_series size is: {}'.format(len(sorted_user_series)))
        # user_index2item = sorted_user_series.keys().tolist()  # 所有用户，按照从高到低排序
        user_index2item = sorted_user_series.head(top_n_user).keys().tolist()  # 所有用户，按照从高到低排序

        print('Build index2item')
        sorted_item_series = data.groupby(['tran_name_id']).size().sort_values(ascending=False)
        print('sorted_item_series size is: {}'.format(len(sorted_item_series)))
        # item_index2item = sorted_item_series.head(top_n_item).keys().tolist()  # 只取前 top_n个item
        item_index2item = []
        for item_id, count in sorted_item_series.iteritems():
            if count >= freq_n_item:
                item_index2item.append(item_id)
            else:
                break
        print('item_index2item size is: {}'.format(len(item_index2item)))#item数目

        # -------------------------------------------------------------------------------------
        # 加入音乐集对应作者的映射,对于topN对应不同的编码
        # 1.取出所有对应的作者集合 2.根据作者集合对作者进行编码 3.根据音乐-作者的映射构建字典
        music2artist_all = data.set_index("tran_name_id")["art_name_id"].to_dict()
        artist_list = [music2artist_all[x] for x in item_index2item]
        artist_list = np.unique(artist_list)  # 去重
        # 音乐id：0 ≤ index ＜ top_n_item
        # 艺术家id：top_n_ite ≤ index
        print('the number of artists:{}'.format(len(artist_list)))
        artistID = dict((v, i + len(item_index2item)) for i, v in enumerate(artist_list))
        music2artist_res = dict()
        for i, v in enumerate(item_index2item):
            music2artist_res[i] = artistID[music2artist_all[v]]
        music2artistFile = os.path.join(BASE_DIR, DATA_SOURCE,  'last_music_index2artist_freq' + str(freq_n_item) + '_topu' + str(top_n_user))
        pickle.dump(music2artist_res, open(music2artistFile, 'wb'))

        #音乐映射到album上，判断字典中是否有key，没有对应的album映射为-1，在构建图的时候加入判断
        name2mbid_dic = data.set_index("tran_name_id")["traid"].to_dict()
        musicInfo = json.load(open(os.path.join(BASE_DIR,'lastfm_export_data/track_info.json')))
        music2album = dict()#mbid→音乐名
        for m in musicInfo:
            # 1.音乐不存在mbid 2.音乐不存在专辑 3.音乐专辑不存在mbid
            # 没有mbid的音乐直接删除，没有album的音乐映射为
            trackInfo = m['track']
            if 'mbid' not in trackInfo:  # 音乐没有mbid
                continue
            mbid = m['track']['mbid']
            if 'album' not in trackInfo:
                continue
            album_name_id = m['track']['album']['title']
            music2album[mbid] = album_name_id
        album_list=[]#过滤不存在对应的album0
        for x in item_index2item:
            mbid=name2mbid_dic[x]
            if mbid not in music2album:
                continue
            album_list.append(music2album[mbid])
        album_list=np.unique(album_list)
        print('the number of albums:{}'.format(len(album_list)))
        # name→id
        albumID = dict((v, i + len(item_index2item)+len(artist_list)) for i, v in enumerate(album_list))
        music2album_res = dict()
        cnt=0#统计有对应专辑的个数
        for i, v in enumerate(item_index2item):
            mbid=name2mbid_dic[v]
            if mbid not in music2album:
                music2album_res[i]=-1#不存在对应的专辑
            else:
                music2album_res[i] = albumID[music2album[mbid]]
                cnt+=1
        print("the number of song with album:{}".format(cnt))
        music2albumFile = os.path.join(BASE_DIR, DATA_SOURCE,
                                        'last_music_index2album_freq' + str(freq_n_item) + '_topu' + str(top_n_user))
        pickle.dump(music2album_res, open(music2albumFile, 'wb'))

        #-------------------------------------------------------------------------------------

        # new_user_index2item = [('user_' + str(x)) for x in user_index2item]  # 区分用户和item
        # index2item = item_index2item + new_user_index2item  # user和item都放在index2item里面
        index2item = item_index2item + user_index2item  # user和item都放在index2item里面
        print('index2item size is: {}'.format(len(index2item)))

        print('Most common song is "%s":%d' % (index2item[0], sorted_item_series[0]))
        print('Most active user is "%s":%d' % (index2item[len(item_index2item)], sorted_user_series[0]))

        print('build item2index')
        item2index = dict((v, i) for i, v in enumerate(index2item))

        pickle.dump(index2item, open(index2item_path, 'wb'))
        pickle.dump(item2index, open(item2index_path, 'wb'))

    if os.path.exists(item_length_list_dict_path):
        item_length_list_dict = pickle.load(open(item_length_list_dict_path, 'rb'))
        if os.path.exists(item_length_path):
            item_length = pickle.load(open(item_length_path, 'rb'))
        else:
            print('Build item_length')
            item_length = get_item_length(item_length_list_dict, item2index)
            pickle.dump(item_length, open(item_length_path, 'wb'))
    else:
        print('Build item_length_list_dict and item_length')
        item_length_list_dict = get_item_lengths_list(data)
        item_length = get_item_length(item_length_list_dict, item2index)

        pickle.dump(item_length, open(item_length_path, 'wb'))
        pickle.dump(item_length_list_dict, open(item_length_list_dict_path, 'wb'))

    if not os.path.exists(index2length_path):
        print('Build index2length')
        index2length_file = open(index2length_path, 'w', encoding='utf-8')
        for item in index2item:
            if item in item_length:
                index2length_file.write(str(item_length[item]) + '\n')
        index2length_file.close()

    if not os.path.exists(index2words_path):
        print('Build index2words')
        item2words_path = os.path.join(BASE_DIR, DATA_SOURCE, 'id_words.txt')
        if not os.path.exists(item2words_path):
            item2words_file = open(item2words_path, 'w', encoding='utf-8')
            tran_group = data.groupby(['tran_name_id'])
            total = len(tran_group)
            count = 0
            print("start processing")
            for tran_name_id, length in tran_group.size().sort_values().iteritems():
                tran_data = tran_group.get_group(tran_name_id)
                art_name_id = tran_data['art_name_id'].tolist()[0].replace(" ", "")
                item2words_file.write(tran_name_id + "~=~=~" + art_name_id + "\n")
                if count % 100000 == 0:
                    print("=====count %d/%d======" % (count, total))
                count += 1
            item2words_file.close()
        index2words_file = open(index2words_path, 'w', encoding='utf-8')
        item_words_dict = dict()
        for line in open(item2words_path, encoding='utf-8'):
            item, words = line.strip().split("~=~=~")
            item_words_dict[item] = words
        print(len(item_words_dict))
        for item in index2item:
            if item in item_words_dict:
                index2words_file.write(item_words_dict[item] + '\n')
            elif not item.startswith("user_"):
                index2words_file.write('\n')
            else:
                # print("empty id is {}".format(item))
                pass
        index2words_file.close()

    print('start loop')
    count = 0
    valid_user_count = 0
    user_group = data.groupby(['user_id'])
    total = len(user_group)
    # short sequence comes first
    for user_id, length in user_group.size().sort_values().iteritems():
        if small_data and count == 100:
            break

        if count % 100 == 0:
            print("=====count %d/%d======" % (count, total))
            print('%s %d' % (user_id, length))
        count += 1

        if user_id not in item2index:
            continue

        # oldest data comes first
        temp_user_data = user_group.get_group(user_id)
        old_time_seq = copy.deepcopy(pd.to_datetime(temp_user_data['timestamp']))
        temp_user_data.loc[:, 'timestamp_new'] = old_time_seq
        user_data = temp_user_data.sort_values(by='timestamp_new')
        # user_data = user_data[user['tranname'].notnull()]
        music_seq = user_data['tran_name_id']
        time_seq = user_data['timestamp_new']
        # filter the null data.
        # time_seq = time_seq[time_seq.notnull()]
        # 应该根据相同的标准进行过滤
        time_seq = time_seq[music_seq.notnull()]
        time_seq_list = time_seq.tolist()
        music_seq = music_seq[music_seq.notnull()]

        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = time_seq.diff(-1).astype('timedelta64[s]') * -1
        # map music to index
        item_seq_list = music_seq.apply(lambda x: (item2index[x]) if pd.notnull(x) and x in item2index else -1).tolist()

        # 因为delta_time[-1] 是nan
        delta_time_list = delta_time.tolist()
        delta_time_list[-1] = 0

        length_time_list = music_seq.apply(
            lambda x: (item_length[x]) if pd.notnull(x) and x in item_length else -1).tolist()

        # 对数据序列进行进一步处理，中间出现了很多计算错误：
        # 例如同一首歌连续出现，但是每次的间隔时间为0秒、1秒、2秒
        valid_index = [0]
        need_record = False
        temp_sum = 0
        for i in range(1, len(item_seq_list)):
            # if item_seq[i] != item_seq[i-1] or delta_time[i-1] >= int(length_time[i-1])/2:
            #     valid_index.append(i)
            if item_seq_list[i] != item_seq_list[i - 1]:
                valid_index.append(i)
                temp_sum = delta_time_list[i]
                if temp_sum >= length_time_list[i]:
                    need_record = True
                else:
                    need_record = False
            else:
                if need_record:
                    valid_index.append(i)
                    temp_sum = delta_time_list[i]
                else:
                    temp_sum += delta_time_list[i]

                if temp_sum >= length_time_list[i]:
                    need_record = True
                else:
                    need_record = False

        temp_item_seq = []
        temp_length_time = []
        temp_time_seq = []
        for i in range(len(valid_index)):
            index = valid_index[i]
            temp_item_seq.append(item_seq_list[index])
            temp_length_time.append(length_time_list[index])
            temp_time_seq.append(time_seq_list[index])
        # temp_delta_time = np.diff(temp_time_seq).tolist()
        temp_delta_time = (pd.Series(temp_time_seq).diff(-1).astype('timedelta64[s]') * -1).tolist()
        temp_delta_time.append(0)

        # if NORM_METHOD == 'log':
        #     # 这里我们使用对数函数来对间隔时间进行缩放
        #     # + 1.0 + 1e-6  保证结果为正数
        #     delta_time = np.log(np.array(delta_time) + 1.0 + 1e-6)  # log不写底数时默认以e为底
        #     length_time = np.log(np.array(length_time) + 1.0 + 1e-6)  # log不写底数时默认以e为底
        # elif NORM_METHOD == 'mm':
        #     temp_delta_time = np.array(delta_time)
        #     min_delta = temp_delta_time.min()
        #     max_delta = temp_delta_time.max()
        #     # (temp_delta_time - min_time) / (max_time - min_time)
        #     delta_time = (np.array(delta_time) - min_delta) / (max_delta - min_delta)
        #
        #     temp_length_time = np.array(length_time)
        #     min_delta = temp_length_time.min()
        #     max_delta = temp_length_time.max()
        #     length_time = (np.array(length_time) - min_delta) / (max_delta - min_delta)
        # elif NORM_METHOD == 'hour':
        if NORM_METHOD == 'hour':
            temp_delta_time = np.array(temp_delta_time) / 3600
            temp_length_time = np.array(temp_length_time) / 3600

        time_accumulate = [0]
        # 将a和b之间的间隔，放到b上，便于计算后面的时间戳 time_accumulate
        for delta in temp_delta_time[:-1]:
            # if delta == 0.0 or delta ==-0.0:
            #     print("delta: {}".format(delta))
            next_time = time_accumulate[-1] + delta
            time_accumulate.append(next_time)

        # 过滤-1的歌曲
        new_item_seq = []
        new_time_accumulate = []
        new_length_time = []
        new_delta_time = []
        valid_count = 0
        for i in range(len(temp_item_seq)):  # 过滤掉 -1 的item
            # if item_seq[i] != -1:
            # 避免delta_time==0
            if temp_item_seq[i] != -1 and temp_delta_time[i] != 0.0 and temp_delta_time[i] != -0.0:
                new_item_seq.append(temp_item_seq[i])
                new_time_accumulate.append(time_accumulate[i])
                new_length_time.append(temp_length_time[i])
                new_delta_time.append(temp_delta_time[i])
                valid_count += 1
            if valid_count >= max_length:
                break

        if len(new_item_seq) < min_length:  # 跳过过滤之后的交互记录少于min_length的用户
            continue
        else:
            valid_user_count += 1
            user_index = item2index[user_id]
            # baseline的用户index从0开始
            user_index_baseline = user_index - baseline_config.top_n_item_dict_lastfm[freq_n_item]
            # 在 hash() 对对象使用时，所得的结果不仅和对象的内容有关，还和对象的 id()，也就是内存地址有关。
            # --------------------------------------------------------------------------------------------
            # 修改为对全部usr都进行划分
            pre_index = int(len(new_item_seq) * partition)
            for i in range(pre_index):  # 前50%序列，训练
                out_tr_uit.write(
                    str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                        new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')
            for i in range(pre_index, int(len(new_item_seq))):  # 后50%序列，测试
                out_te_old_uit.write(
                    str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                        new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')
                out_te_new_uit.write(
                    str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                        new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')

            # baseline
            out_tr_baseline_uit.write(str(user_index_baseline) + ',')
            out_tr_baseline_uit.write(':'.join(str(x) for x in new_item_seq[:pre_index]) + '\n')

            out_te_baseline_old_uit.write(str(user_index_baseline) + ',')
            out_te_baseline_old_uit.write(':'.join(str(x) for x in new_item_seq[pre_index:]) + '\n')

            out_te_baseline_new_uit.write(str(user_index_baseline) + ',')
            out_te_baseline_new_uit.write(':'.join(str(x) for x in new_item_seq[pre_index:]) + '\n')

            # lstm baseline
            out_tr_lstm_uit.write(str(user_index_baseline) + ',')
            out_tr_lstm_uit.write(':'.join(str(x) for x in new_item_seq[:pre_index]) + '\n')

            out_te_lstm_old_uit.write(str(user_index_baseline) + ',')
            out_te_lstm_old_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

            out_te_lstm_new_uit.write(str(user_index_baseline) + ',')
            out_te_lstm_new_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

    print("valid_user_count is: {}".format(valid_user_count))
    out_tr_uit.close()
    out_te_old_uit.close()
    out_te_new_uit.close()
    out_tr_baseline_uit.close()
    out_te_baseline_old_uit.close()
    out_te_baseline_new_uit.close()
    out_tr_lstm_uit.close()
    out_te_lstm_old_uit.close()
    out_te_lstm_new_uit.close()

if __name__ == '__main__':
    # user, item, timestamp, delta（该item到下一个item的间隔）, length（该item的实际长度）
    BASE_DIR = '/home/wh/MusicSeq/datasets'
    DATA_SOURCE = 'lastfm'

    # top_n = 5000  # 频率排前topn的音乐
    # top_n_item = 10000  # 频率排前topn的音乐
    min_length = 100  # more than
    # max_length = 200  # more than
    # max_length = 1000  # more than
    max_length = 1500  # more than
    # top_n_user = 1000  # 频率排前topn的user
    top_n_user = 900  # 频率排前topn的user
    # 80%用户的所有序列作为训练；
    # 20%用户的前50%序列训练，后50%作为next item测试；
    # 同样的20%用户的前50%序列训练，后50%作为next new item测试；
    # 对于给定的测试用户u（n次交互记录），则会产生n-1个测试用例
    # 对间隔时间进行标准化/归一化
    # 由于每个用户有自己的衰减稀疏delta，所以可以分别对每个用户的数据进行归一化

    path = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname.tsv')
    # path = os.path.join(BASE_DIR, DATA_SOURCE, 'temp.txt')

    print("start reading csv")
    # data = pd.read_csv(path, sep='\t',
    #                    error_bad_lines=False,
    #                    header=None,
    #                    names=['user_id', 'timestamp', 'artid', 'artname', 'traid', 'tranname'])
    # quotechar=None, quoting=3 避免导致错误的字符 "
    data = pd.read_csv(path, sep='\t',
                       error_bad_lines=False,
                       header=None,
                       names=['user_id', 'timestamp', 'artid', 'artname', 'traid', 'tranname'],
                       quotechar=None, quoting=3)
    print("finish reading csv")
    # 避免重名的歌曲和歌手
    data['tran_name_id'] = data['tranname'] + data['traid']
    data['art_name_id'] = data['artname'] + data['artid']
    # for top_n_item in [10000, 9000, 8000, 7000, 6000, 5000]:
    # for top_n_item in [10000]:
    # for freq_n_item in [10000]:
    # for freq_n_item in [50, 100, 150, 200, 250]:

    #调整过滤频率
    freq_n_item=50
    for partition in [0.1, 0.2, 0.3, 0.4]:
    #for partition in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print("starting processing for freq_n_item = {},partition = {}".format(freq_n_item,partition))
        generate_data(freq_n_item, top_n_user, min_length, max_length, data, BASE_DIR, DATA_SOURCE, partition,small_data=False)
