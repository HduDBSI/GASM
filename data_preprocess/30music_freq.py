# -*- coding: utf-8  -*-
from __future__ import print_function
import pandas as pd
import pickle
import os
import numpy as np
import copy
from collections import defaultdict
import json
import baseline_config


def get_item_lengths_list(data):
    item_length_list_dict = dict()
    user_group = data.groupby(['user_id'])
    # short sequence comes first
    for user_id, length in user_group.size().sort_values().iteritems():
        # oldest data comes first
        # user_data = user_group.get_group(user_id).sort_values(by='timestamp')
        user_data = user_group.get_group(user_id).sort_values(by=['timestamp', 'event_id'])
        # user_data = user_data[user['tranname'].notnull()]
        music_seq = user_data['song_id']
        time_seq = user_data['timestamp']
        length_seq = user_data['song_length']
        # filter the null data.
        # time_seq = time_seq[time_seq.notnull()]
        # 应该根据相同的标准进行过滤，而且是先过滤其他的，再过滤music_seq
        time_seq = time_seq[music_seq.notnull()]
        length_seq = length_seq[music_seq.notnull()]
        music_seq = music_seq[music_seq.notnull()]

        # 对数据序列进行进一步处理，中间出现了很多计算错误：
        # 例如同一首歌连续出现，但是每次的间隔时间为0秒、1秒、2秒

        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = time_seq.diff(-1) * -1
        item_seq = music_seq.tolist()
        length_seq = length_seq.tolist()
        delta_time = delta_time.tolist()
        delta_time[-1] = 0  # 原本的delta_time[-1] 是 nan

        for item, delta, song_length in zip(item_seq, delta_time, length_seq):
            if item in item_length_list_dict:
                length_list_temp = item_length_list_dict.get(item)
            else:
                length_list_temp = []
            if int(song_length) > 0:  # 优先按照长度记录，如果为-1，则按照间隔时间计算
                length_list_temp.append(int(song_length))
            else:
                length_list_temp.append(int(delta))
            item_length_list_dict[item] = length_list_temp
    return item_length_list_dict


def get_item_length(item_length_list_dict, item2index):
    item_length = dict()
    # short sequence comes first
    for item in item_length_list_dict.keys():
        if item in item2index:
            length_list = item_length_list_dict[item]
            max_length = max(length_list, key=length_list.count)
            # if length_list.count(max_length) >= 2 and max_length < 3600:
            if length_list.count(max_length) >= 2:
                item_length[item] = max_length
            else:
                print(item + ": ")
                print(length_list)
                # item_length[item] = -1
    return item_length


def generate_data(freq_n_item, top_n_user, min_length, max_length, data, BASE_DIR, DATA_SOURCE,partition):
    # NORM_METHOD = 'origin'

    # NORM_METHOD = 'log'
    # (x-min)/(max-min)
    # NORM_METHOD = 'mm'
    # /3600
    NORM_METHOD = 'hour'

    tr_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                            'tr-user-item-time-freq{}-min{}-max{}-{}-{}'.format(freq_n_item, min_length,
                                                                                                max_length,
                                                                                                NORM_METHOD,
                                                                                                int(partition * 10)) + '.lst')
    # next item
    te_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                'te-user-old-item-time-freq{}-min{}-max{}-{}-{}'.format(freq_n_item,
                                                                                                        min_length,
                                                                                                        max_length,
                                                                                                        NORM_METHOD,
                                                                                                        int(partition * 10)) + '.lst')
    # next new item
    te_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                'te-user-new-item-time-freq{}-min{}-max{}-{}-{}'.format(freq_n_item,
                                                                                                        min_length,
                                                                                                        max_length,
                                                                                                        NORM_METHOD,
                                                                                                        int(partition * 10)) + '.lst')
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
    tr_lstm_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                 'lstm_tr_freq{}_partition{}.lst'.format(freq_n_item,
                                                                                         int(partition * 10)))
    te_lstm_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                     'lstm_te_old_freq{}_partition{}.lst'.format(freq_n_item,
                                                                                                 int(partition * 10)))
    te_lstm_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                     'lstm_te_new_freq{}_partition{}.lst'.format(freq_n_item,
                                                                                                 int(partition * 10)))

    index2item_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   '30music_index2item_freq' + str(freq_n_item) + '_topu' + str(top_n_user))
    item2index_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   '30music_item2index_freq' + str(freq_n_item) + '_topu' + str(top_n_user))
    index2length_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                     '30music_index2length_freq' + str(freq_n_item) + '_topu' + str(top_n_user))
    index2words_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                    '30music_index2words_freq' + str(freq_n_item) + '_topu' + str(top_n_user))

    item_length_path = os.path.join(BASE_DIR, DATA_SOURCE, '30music_item_length_freq' + str(freq_n_item))
    item_length_list_dict_path = os.path.join(BASE_DIR, DATA_SOURCE, '30music_item_length_list_dict')
    # -------------------------------------------------------------------------------------
    out_tr_uit = open(tr_user_item_time_record, 'w', encoding='utf-8')
    out_te_old_uit = open(te_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_new_uit = open(te_user_new_item_time_record, 'w', encoding='utf-8')
    
    out_tr_baseline_uit = open(tr_baseline_user_item_time_record, 'w', encoding='utf-8')
    # 首行
    out_tr_baseline_uit.write(str(top_n_user) + ", " + str(baseline_config.top_n_item_dict_30music[freq_n_item]) + '\n')
    
    out_te_baseline_old_uit = open(te_baseline_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_baseline_new_uit = open(te_baseline_user_new_item_time_record, 'w', encoding='utf-8')
    
    # lstm baseline
    out_tr_lstm_uit = open(tr_lstm_user_item_time_record, 'w', encoding='utf-8')
    # 首行
    out_tr_lstm_uit.write(str(top_n_user) + ", " + str(baseline_config.top_n_item_dict_30music[freq_n_item]) + '\n')
    
    out_te_stm_old_uit = open(te_lstm_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_stm_new_uit = open(te_lstm_user_new_item_time_record, 'w', encoding='utf-8')
    # -------------------------------------------------------------------------------------
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
        sorted_item_series = data.groupby(['song_id']).size().sort_values(ascending=False)
        print('sorted_item_series size is: {}'.format(len(sorted_item_series)))
        # item_index2item = sorted_item_series.head(freq_n_item).keys().tolist()  # 只取前 top_n个item
        item_index2item = []
        for item_id, count in sorted_item_series.iteritems():
            if count >= freq_n_item:
                item_index2item.append(item_id)
            else:
                break
        print('item_index2item size is: {}'.format(len(item_index2item)))

        # -------------------------------------------------------------------------------------
        musicInfo_file_path = '../datasets/30Music/entities/tracks.idomaar'
        musicInfo_df = pd.read_csv(musicInfo_file_path, sep='\t',
                                   error_bad_lines=False,
                                   header=None,
                                   index_col=False,
                                   names=['type', 'song_id', 'flag', 'song_record', 'song_info'])#5675143条数据
        music2artist_dict = dict()  # id-id
        music2album_dict = dict()
        artist_list, album_list = [], []
        item_index2item_set = set(item_index2item)#使用set加速查询速度

        music_list=musicInfo_df['song_id']
        musicInfo_list=musicInfo_df['song_info']

        for m, info in zip(music_list,musicInfo_list):
            # print(index,row)
            musicId = 'i_' + str(m)
            if musicId not in item_index2item_set:  # 剔除无关音乐对应的信息
                continue
            musicInfo_dict_slice = json.loads(info)
            # artist必定存在，默认最多一个artist和album
            artistId = musicInfo_dict_slice['artists'][0]['id']

            music2artist_dict[musicId] = (artistId)
            artist_list.append(artistId)
            if musicInfo_dict_slice['albums']!=None and len(musicInfo_dict_slice['albums']) != 0 and musicInfo_dict_slice['albums'][0]['id'] != None:  # 不存在album
                albumId = musicInfo_dict_slice['albums'][0]['id']
                album_list.append(albumId)
                music2album_dict[musicId] = albumId
            #如果不存在不需要加入字典，否则list中要加入-1

        artist_list = np.unique(artist_list)
        album_list = np.unique(album_list)

        print('the number of artists:{}'.format(len(artist_list)))
        print('the number of albums:{}'.format(len(album_list)))

        # 音乐信息-id
        artistID = dict((v, i + len(item_index2item)) for i, v in enumerate(artist_list))
        albumID = dict((v, i + len(item_index2item) + len(artist_list)) for i, v in enumerate(album_list))
        # 输出字典 item下标-item下标
        music2artist_res = dict()
        music2album_res = dict()
        # 维护两个musicId的set，判断对应的item有无信息
        music_in_artist_set = set(music2artist_dict.keys())
        music_in_album_set = set(music2album_dict.keys())
        music_noInfo_list = []
        for i, v in enumerate(item_index2item):
            if v not in music_in_artist_set and v not in music_in_album_set:
                # 两种信息都没有说明不存在附加信息
                music_noInfo_list.append(v)
                music2artist_res[i] = -1
                music2album_res[i] = -1
            elif v not in music_in_artist_set:
                music2artist_res[i] = -1
                music2album_res[i] = albumID[music2album_dict[v]]
            elif v not in music_in_album_set:
                music2artist_res[i] = artistID[music2artist_dict[v]]
                music2album_res[i] = -1
            else:
                music2album_res[i] = albumID[music2album_dict[v]]
                music2artist_res[i] = artistID[music2artist_dict[v]]

        print("the freq of music without info：{}".format(len(music_noInfo_list)))
        music_noInfo_list = np.unique(music_noInfo_list)
        print("the number of music without info：{}".format(len(music_noInfo_list)))

        music2artistFile = os.path.join(BASE_DIR, DATA_SOURCE,
                                        '30Music_music_index2artist_freq' + str(freq_n_item) + '_topu' + str(
                                            top_n_user))
        pickle.dump(music2artist_res, open(music2artistFile, 'wb'))
        music2albumFile = os.path.join(BASE_DIR, DATA_SOURCE,
                                       '30Music_music_index2album_freq' + str(freq_n_item) + '_topu' + str(top_n_user))
        pickle.dump(music2album_res, open(music2albumFile, 'wb'))

        # -------------------------------------------------------------------------------------
        # 拼接在后面不影响前面的id
        new_user_index2item = [str(x) for x in user_index2item]  # 区分用户和item
        index2item = item_index2item + new_user_index2item
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

    # 不存在对应的id_words文件
    if 0 and not os.path.exists(index2words_path):
        print('Build index2words')
        item2words_path = os.path.join(BASE_DIR, DATA_SOURCE, 'id_words.txt')
        index2words_file = open(index2words_path, 'w', encoding='utf-8')
        item_words_dict = dict()
        for line in open(item2words_path, encoding='utf-8'):
            item, words = line.strip().split("~=~=~")
            item_words_dict[item] = words
        print(len(item_words_dict))
        for item in index2item:
            if item in item_words_dict:
                index2words_file.write(item_words_dict[item] + '\n')
            elif item.startswith("s_"):
                index2words_file.write('\n')
            else:
                print("empty id is {}".format(item))
                pass
        index2words_file.close()
    print('=========================')
    #print(item_length)
    print('=========================')
    #
    # length_count_dict = {}
    # for item in item_length:
    #     length = int(item_length[item] / 10) * 10
    #     count = length_count_dict.get(length, 0) + 1
    #     length_count_dict[length] = count
    #
    # print(len(length_count_dict))
    # print(length_count_dict)
    #
    # pic_length = 20
    # pic_height = 10
    # plt.figure(figsize=(pic_length, pic_height))
    # for key in length_count_dict:  # math.log(key,10)  math.log(weight_count_dict1[key],10)
    #     plt.scatter(key, length_count_dict[key])
    # plt.show()

    print('start loop')
    count = 0
    valid_user_count = 0
    user_group = data.groupby(['user_id'])
    total = len(user_group)
    # short sequence comes first
    for user_id, length in user_group.size().sort_values().iteritems():
        if count % 1000 == 0:
            print("=====count %d/%d======" % (count, total))
            print('%s %d' % (user_id, length))
        count += 1

        if user_id not in item2index:
            continue
        # oldest data comes first
        # user_data = user_group.get_group(user_id).sort_values(by='timestamp')
        # 如果出现重复的时间戳，则按照event_id进行排序
        user_data = user_group.get_group(user_id).sort_values(by=['timestamp', 'event_id'])
        # user_data = user_data[user['tranname'].notnull()]
        music_seq = user_data['song_id']
        time_seq = user_data['timestamp']
        # 应该根据相同的标准进行过滤
        time_seq = time_seq[music_seq.notnull()]
        time_seq_list = time_seq.tolist()
        music_seq = music_seq[music_seq.notnull()]
        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = time_seq.diff(-1) * -1
        # map music to index
        item_seq_list = music_seq.apply(lambda x: (item2index[x]) if pd.notnull(x) and x in item2index else -1).tolist()
        # length_seq = length_seq.tolist()

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
        temp_delta_time = np.diff(temp_time_seq).tolist()
        temp_delta_time.append(0)

        # if NORM_METHOD == 'log':
        #     # 这里我们使用对数函数来对间隔时间进行缩放
        #     # + 1.0 + 1e-6  保证结果为正数
        #     temp_delta_time = np.log(np.array(temp_delta_time) + 1.0 + 1e-6)  # log不写底数时默认以e为底
        #     temp_length_seq = np.log(np.array(temp_length_seq) + 1.0 + 1e-6)  # log不写底数时默认以e为底
        # elif NORM_METHOD == 'mm':
        #     temp_delta_time = np.array(temp_delta_time)
        #     min_delta = temp_delta_time.min()
        #     max_delta = temp_delta_time.max()
        #     # (temp_delta_time - min_time) / (max_time - min_time)
        #     temp_delta_time = (np.array(temp_delta_time) - min_delta) / (max_delta - min_delta)
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
        for i in range(len(temp_item_seq)):
            # for i in valid_index:
            # if item_seq[i] != -1:  # 过滤掉 -1 的item
            # 避免delta_time==0
            if temp_item_seq[i] != -1 and temp_delta_time[i] != 0.0 and temp_delta_time[i] != -0.0 and temp_length_time[
                i] != -1:
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
            user_index_baseline = user_index - baseline_config.top_n_item_dict_30music[freq_n_item]
            # 在 hash() 对对象使用时，所得的结果不仅和对象的内容有关，还和对象的 id()，也就是内存地址有关。
            # --------------------------------------------------------------------------------------------
            # 修改为对全部usr都进行划分
            pre_index = int(len(new_item_seq) * partition)
            for i in range(pre_index):
                out_tr_uit.write(
                    str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                        new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')
            for i in range(pre_index, int(len(new_item_seq))):
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

            out_te_stm_old_uit.write(str(user_index_baseline) + ',')
            out_te_stm_old_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

            out_te_stm_new_uit.write(str(user_index_baseline) + ',')
            out_te_stm_new_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

    print("valid_user_count is: {}".format(valid_user_count))
    out_tr_uit.close()
    out_te_old_uit.close()
    out_te_new_uit.close()
    out_tr_baseline_uit.close()
    out_te_baseline_old_uit.close()
    out_te_baseline_new_uit.close()


if __name__ == '__main__':
    # user, item, timestamp, delta（该item到下一个item的间隔）, length（该item的实际长度）
    BASE_DIR = '/home/wh/MusicSeq/datasets'
    DATA_SOURCE = '30Music'

    # top_n = 5000  # 频率排前topn的音乐
    # top_n_item = 10000  # 频率排前topn的音乐
    min_length = 100  # more than
    # max_length = 200  # more than
    # max_length = 1000  # more than
    max_length = 1500  # more than
    # user: 45175
    # item: 4519104
    top_n_user = 3000  # 频率排前topn的user
    path = os.path.join(BASE_DIR, DATA_SOURCE, '30music_record.csv')
    print("start reading csv")

    # te_user_item_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te_user_item.lst')

    data = pd.read_csv(path, sep='\t',
                       error_bad_lines=False,
                       header=None,
                       index_col=False,
                       names=['user_id', 'song_id', 'timestamp', 'song_length', 'event_id'],
                       quotechar=None, quoting=3)
    print("finish reading csv")

    # sorted_item_series = data.groupby(['song_id']).size().sort_values(ascending=False)
    # max_count = 1000
    # count_list = np.zeros(max_count)
    # for item_id, count in sorted_item_series.iteritems():
    #     if count >= max_count:
    #         count = max_count-1
    #     count_list[:count] += 1
    # print("=========== {} ===========".format(1))
    # print(count_list[0])
    # for i in range(10, 310 ,10):
    #     print("=========== {} ===========".format(i))
    #     print(count_list[i-1])
    # for top_n_item in [10000, 9000, 8000, 7000, 6000, 5000]:
    # for top_n_item in [20000]:
    # for freq_n_item in [50, 100, 150, 200, 250]:
    freq_n_item=50
    for partition in [0.1, 0.2, 0.3, 0.4]:
    #for partition in [0.5,0.6,0.7,0.8,0.9]:
        print("starting processing for freq_n_item = {},partition = {}".format(freq_n_item, partition))
        generate_data(freq_n_item, top_n_user, min_length, max_length, data, BASE_DIR, DATA_SOURCE,partition)
