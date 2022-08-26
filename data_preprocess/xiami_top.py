# -*- coding: utf-8  -*
from __future__ import print_function
import pandas as pd
import pickle
import os

import numpy as np
import copy


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
        music_seq = user_data['song_id']
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
        delta_time[-1] = 0 # 原本的delta_time[-1] 是 nan
        for item, time in zip(item_seq, delta_time):
            if item in item_length_list_dict:
                length_list_temp = item_length_list_dict.get(item)
            else:
                length_list_temp = []
            length_list_temp.append(int(time))
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
                print(item + ": " + length_list)
                # item_length[item] = -1
    return item_length


def generate_data(top_n_item, top_n_user, min_length, max_length, data, BASE_DIR, DATA_SOURCE):
    # NORM_METHOD = 'origin'

    # NORM_METHOD = 'log'
    # (x-min)/(max-min)
    # NORM_METHOD = 'mm'
    # /3600
    NORM_METHOD = 'hour'

    tr_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                            'tr-user-item-time-top{}-min{}-max{}-{}'.format(top_n_item, min_length,
                                                                                            max_length,
                                                                                            NORM_METHOD) + '.lst')
    # next item
    te_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                'te-user-old-item-time-top{}-min{}-max{}-{}'.format(top_n_item,
                                                                                                    min_length,
                                                                                                    max_length,
                                                                                                    NORM_METHOD) + '.lst')
    # next new item
    te_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                'te-user-new-item-time-top{}-min{}-max{}-{}'.format(top_n_item,
                                                                                                    min_length,
                                                                                                    max_length,
                                                                                                    NORM_METHOD) + '.lst')

    index2item_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   'xiami_music_index2item_top' + str(top_n_item) + '_topu' + str(top_n_user))
    item2index_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                   'xiami_music_item2index_top' + str(top_n_item) + '_topu' + str(top_n_user))
    index2length_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                     'xiami_music_index2length_top' + str(top_n_item) + '_topu' + str(top_n_user))
    index2words_path = os.path.join(BASE_DIR, DATA_SOURCE,
                                    'xiami_music_index2words_top' + str(top_n_item) + '_topu' + str(top_n_user))

    item_length_path = os.path.join(BASE_DIR, DATA_SOURCE, 'xiami_music_item_length_top' + str(top_n_item))
    item_length_list_dict_path = os.path.join(BASE_DIR, DATA_SOURCE, 'xiami_music_item_length_list_dict')

    out_tr_uit = open(tr_user_item_time_record, 'w', encoding='utf-8')
    out_te_old_uit = open(te_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_new_uit = open(te_user_new_item_time_record, 'w', encoding='utf-8')

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
        item_index2item = sorted_item_series.head(top_n_item).keys().tolist()  # 只取前 top_n个item
        print('item_index2item size is: {}'.format(len(item_index2item)))

        new_user_index2item = [str(x) for x in user_index2item]  # 区分用户和item
        index2item = item_index2item + new_user_index2item  # user和item都放在index2item里面
        print('index2item size is: {}'.format(len(index2item)))

        print('Most common song is "%s":%d' % (index2item[0], sorted_item_series[0]))
        print('Most active user is "%s":%d' % (index2item[top_n_item], sorted_user_series[0]))

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
    # print('=========================')
    # print(item_length)
    # print('=========================')
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
        if count % 100 == 0:
            print("=====count %d/%d======" % (count, total))
            print('%s %d' % (user_id, length))
        count += 1
        if user_id not in item2index:
            continue
        # oldest data comes first
        temp_user_data = user_group.get_group(user_id)
        temp_time_seq = copy.deepcopy(pd.to_datetime(temp_user_data['timestamp']))
        temp_user_data.loc[:, 'timestamp_new'] = temp_time_seq
        user_data = temp_user_data.sort_values(by='timestamp_new')
        # user_data = user_data[user['tranname'].notnull()]
        music_seq = user_data['song_id']
        time_seq = user_data['timestamp_new']
        # filter the null data.
        # time_seq = time_seq[time_seq.notnull()]
        # 应该根据相同的标准进行过滤
        time_seq = time_seq[music_seq.notnull()]
        music_seq = music_seq[music_seq.notnull()]

        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = time_seq.diff(-1).astype('timedelta64[s]') * -1
        # map music to index
        item_seq = music_seq.apply(lambda x: (item2index[x]) if pd.notnull(x) and x in item2index else -1).tolist()

        delta_time = delta_time.tolist()
        # 因为delta_time[-1] 是nan
        delta_time[-1] = 0
        length_time = music_seq.apply(lambda x: (item_length[x]) if pd.notnull(x) and x in item_length else -1).tolist()

        if NORM_METHOD == 'log':
            # 这里我们使用对数函数来对间隔时间进行缩放
            # + 1.0 + 1e-6  保证结果为正数
            delta_time = np.log(np.array(delta_time) + 1.0 + 1e-6)  # log不写底数时默认以e为底
            length_time = np.log(np.array(length_time) + 1.0 + 1e-6)  # log不写底数时默认以e为底
        elif NORM_METHOD == 'mm':
            temp_delta_time = np.array(delta_time)
            min_delta = temp_delta_time.min()
            max_delta = temp_delta_time.max()
            # (temp_delta_time - min_time) / (max_time - min_time)
            delta_time = (np.array(delta_time) - min_delta) / (max_delta - min_delta)

            temp_length_time = np.array(length_time)
            min_delta = temp_length_time.min()
            max_delta = temp_length_time.max()
            length_time = (np.array(length_time) - min_delta) / (max_delta - min_delta)
        elif NORM_METHOD == 'hour':
            delta_time = np.array(delta_time) / 3600
            length_time = np.array(length_time) / 3600

        time_accumulate = [0]
        # 将a和b之间的间隔，放到b上，便于计算后面的时间戳 time_accumulate
        for delta in delta_time[:-1]:
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
        for i in range(len(item_seq)):  # 过滤掉 -1 的item
            # if item_seq[i] != -1:
            # 避免delta_time==0
            if item_seq[i] != -1 and delta_time[i] != 0.0 and delta_time[i] != -0.0:
                new_item_seq.append(item_seq[i])
                new_time_accumulate.append(time_accumulate[i])
                new_length_time.append(length_time[i])
                new_delta_time.append(delta_time[i])
                valid_count += 1
            if valid_count >= max_length:
                break

        if len(new_item_seq) < min_length:  # 跳过过滤之后的交互记录少于min_length的用户
            continue
        else:
            valid_user_count += 1
            user_index = item2index[user_id]
            # 在 hash() 对对象使用时，所得的结果不仅和对象的内容有关，还和对象的 id()，也就是内存地址有关。
            index_hash_remaining = user_index % 10
            if index_hash_remaining < 2:  # 20%用户的前50%序列训练，后50%作为测试 （next item 和 next new item）；
                for i in range(int(len(new_item_seq) / 2)):  # 前50%序列，训练
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                            new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')
                for i in range(int(len(new_item_seq) / 2), int(len(new_item_seq))):  # 后50%序列，测试
                    out_te_old_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                            new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')
                    out_te_new_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                            new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')
            else:  # 80%用户的所有序列作为训练
                for i in range(len(new_item_seq)):
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                            new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')
    print("valid_user_count is: {}".format(valid_user_count))
    out_tr_uit.close()
    out_te_old_uit.close()
    out_te_new_uit.close()


if __name__ == '__main__':
    # user, item, timestamp, delta（该item到下一个item的间隔）, length（该item的实际长度）
    BASE_DIR = ''
    DATA_SOURCE = 'xiami_music'

    # top_n = 5000  # 频率排前topn的音乐
    # top_n_item = 10000  # 频率排前topn的音乐
    min_length = 100  # more than
    # max_length = 200  # more than
    # max_length = 1000  # more than
    max_length = 1500  # more than
    top_n_user = 4284  # 频率排前topn的user
    path = os.path.join(BASE_DIR, DATA_SOURCE, 'xiami1000.dat')
    print("start reading csv")

    te_user_item_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te_user_item.lst')

    data = pd.read_csv(path, sep='\t',
                       error_bad_lines=False,
                       header=None,
                       index_col=False,
                       names=['user_id', 'song_id', 'timestamp', 'device', 'song_name', 'artist_name'],
                       quotechar=None, quoting=3)
    print("finish reading csv")

    # for top_n_item in [10000, 9000, 8000, 7000, 6000, 5000]:
    for top_n_item in [20000]:
        print("starting processing for top_n_item = {}".format(top_n_item))
        generate_data(top_n_item, top_n_user, min_length, max_length, data, BASE_DIR, DATA_SOURCE)
