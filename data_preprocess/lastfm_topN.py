# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import pickle
import os

import numpy as np
import copy


# 80%用户的所有序列作为训练；
# 20%用户的前50%序列训练，后50%作为next item测试；
# 同样的20%用户的前50%序列训练，后50%作为next new item测试；
# 对于给定的测试用户u（n次交互记录），则会产生n-1个测试用例
# 对间隔时间进行标准化/归一化
# 由于每个用户有自己的衰减稀疏delta，所以可以分别对每个用户的数据进行归一化

# path = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname.tsv')

# filter bad lines in original file
# path_filtered = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname-filtered.tsv')
# path_filtered = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname-small.tsv')
# path_filtered = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname.tsv')


def generate_data(user_count, top_n_item, min_length, max_length, data, BASE_DIR, DATA_SOURCE, small_data=False):
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

    # normal baseline
    tr_baseline_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                     'baseline_tr_top{}.lst'.format(top_n_item))
    te_baseline_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                         'baseline_te_old_top{}.lst'.format(top_n_item))
    te_baseline_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                         'baseline_te_new_top{}.lst'.format(top_n_item))

    # lstm baseline
    tr_lstm_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'lstm_tr_top{}.lst'.format(top_n_item))
    te_lstm_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'lstm_te_old_top{}.lst'.format(top_n_item))
    te_lstm_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'lstm_te_new_top{}.lst'.format(top_n_item))

    index2item_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_index2item_top' + str(top_n_item))
    item2index_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_item2index_top' + str(top_n_item))
    out_tr_uit = open(tr_user_item_time_record, 'w', encoding='utf-8')
    out_te_old_uit = open(te_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_new_uit = open(te_user_new_item_time_record, 'w', encoding='utf-8')

    # baseline
    out_tr_baseline_uit = open(tr_baseline_user_item_time_record, 'w', encoding='utf-8')
    # 首行
    out_tr_baseline_uit.write(str(user_count) + ", " + str(top_n_item) + '\n')

    out_te_baseline_old_uit = open(te_baseline_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_baseline_new_uit = open(te_baseline_user_new_item_time_record, 'w', encoding='utf-8')

    # lstm baseline
    out_tr_lstm_uit = open(tr_lstm_user_item_time_record, 'w', encoding='utf-8')
    # 首行
    out_tr_lstm_uit.write(str(user_count) + ", " + str(top_n_item) + '\n')

    out_te_lstm_old_uit = open(te_lstm_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_lstm_new_uit = open(te_lstm_user_new_item_time_record, 'w', encoding='utf-8')

    # print("start reading csv")
    # # data = pd.read_csv(path, sep='\t',
    # #                    error_bad_lines=False,
    # #                    header=None,
    # #                    names=['userid', 'timestamp', 'artid', 'artname', 'traid', 'tranname'])
    # # quotechar=None, quoting=3 避免导致错误的字符 "
    # data = pd.read_csv(path, sep='\t',
    #                    error_bad_lines=False,
    #                    header=None,
    #                    names=['userid', 'timestamp', 'artid', 'artname', 'traid', 'tranname'],
    #                    quotechar=None, quoting=3)
    # print("finish reading csv")

    if os.path.exists(index2item_path) and os.path.exists(item2index_path):
        index2item = pickle.load(open(index2item_path, 'rb'))
        item2index = pickle.load(open(item2index_path, 'rb'))
        print('Total music and user %d' % len(index2item))
    else:
        print('Build index2item')
        #根据用户id分组，并求出每个用户的记录数-行数（字典）
        sorted_user_series = data.groupby(['userid']).size().sort_values(ascending=False)
        print('sorted_user_series size is: {}'.format(len(sorted_user_series)))
        user_index2item = sorted_user_series.keys().tolist()  # 所有用户，按照从高到低排序

        #统计音乐出现的频次
        sorted_item_series = data.groupby(['tran_name_id']).size().sort_values(ascending=False)
        print('sorted_item_series size is: {}'.format(len(sorted_item_series)))
        item_index2item = sorted_item_series.head(top_n_item).keys().tolist()  # 只取前 top_n个item
        print('item_index2item size is: {}'.format(len(item_index2item)))

        #加入音乐集对应作者的映射,对于topN对应不同的编码
        #1.取出所有对应的作者集合 2.根据作者集合对作者进行编码 3.根据音乐-作者的映射构建字典
        music2artist_all=data.set_index("tran_name_id")["art_name_id"].to_dict()
        artist_list=[music2artist_all[x] for x in item_index2item]
        artist_list=np.unique(artist_list)#去重
        #音乐id：0 ≤ index ＜ top_n_item
        #艺术家id：top_n_ite ≤ index
        artistID=dict((v, i+top_n_item) for i, v in enumerate(artist_list))
        music2artist_res=dict()
        for i, v in enumerate(item_index2item):
            music2artist_res[i]=artistID[music2artist_all[v]]
        music2artistFile = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_music2artist_top' + str(top_n_item))
        pickle.dump(music2artist_res, open(music2artistFile, 'wb'))

        new_user_index2item = [('user_' + str(x)) for x in user_index2item]  # 区分用户和item
        index2item = item_index2item + new_user_index2item  # user和item都放在index2item里面
        print('index2item size is: {}'.format(len(index2item)))


        print('Most common song is "%s":%d' % (index2item[0], sorted_item_series[0]))
        print('Most active user is "%s":%d' % (index2item[top_n_item], sorted_user_series[0]))

        print('build item2index')
        item2index = dict((v, i) for i, v in enumerate(index2item))# dict根据列表的下标对于用户和音乐进行编码

        pickle.dump(index2item, open(index2item_path, 'wb'))
        pickle.dump(item2index, open(item2index_path, 'wb'))

    print('start loop')
    count = 0
    valid_user_count = 0
    user_group = data.groupby(['userid'])#得到每个用户的df
    total = len(user_group)
    # short sequence comes first
    for user_id, length in user_group.size().sort_values().iteritems():
        if count % 100 == 0:
            print("=====count %d/%d======" % (count, total))
            print('%s %d' % (user_id, length))
        count += 1
        # # oldest data comes first
        # user_data = user_group.get_group(user_id).sort_values(by='timestamp')
        # # user_data = user_data[user['tranname'].notnull()]
        # music_seq = user_data['tranname']
        # time_seq = user_data['timestamp']
        # oldest data comes first
        temp_user_data = user_group.get_group(user_id)
        old_time_seq = copy.deepcopy(pd.to_datetime(temp_user_data['timestamp']))
        temp_user_data.loc[:, 'timestamp_new'] = old_time_seq
        user_data = temp_user_data.sort_values(by='timestamp_new')
        # user_data = user_data[user['tranname'].notnull()]
        music_seq = user_data['tran_name_id']
        time_seq = user_data['timestamp_new']
        # filter the null data.
        time_seq = time_seq[music_seq.notnull()]
        # time_seq_list = time_seq.tolist()
        music_seq = music_seq[music_seq.notnull()]
        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = pd.to_datetime(time_seq).diff(-1).astype('timedelta64[s]') * -1
        # map music to index
        item_seq = music_seq.apply(lambda x: (item2index[x]) if pd.notnull(x) and x in item2index else -1).tolist()

        delta_time = delta_time.tolist()
        # delta_time[-1] = 0
        # delta_time[-1] = 1e-6
        delta_time[-1] = 0

        # temp_log_delta_time = np.log(np.array(delta_time) + 1.0 + 1e-6)  # 不写底数时默认以e为底
        # min_log_delta = temp_log_delta_time.min()
        # max_log_delta = temp_log_delta_time.max()
        # mean_log_delta = temp_log_delta_time.mean()
        # std_log_delta = temp_log_delta_time.std()
        # print("temp_log_delta_time min: {}, max: {}, mean: {}, std: {}".format(min_log_delta, max_log_delta,
        #                                                                        mean_log_delta, std_log_delta))
        # # 归一化
        # temp_delta_time = np.array(delta_time)
        # min_delta = temp_delta_time.min()
        # max_delta = temp_delta_time.max()
        # mean_delta = temp_delta_time.mean()
        # std_delta = temp_delta_time.std()
        # print("temp_delta_time min: {}, max: {}, mean: {}, std: {}".format(min_delta, max_delta, mean_delta, std_delta))

        # temp_delta_time2 = np.array(delta_time)/3600
        # min_delta = temp_delta_time2.min()
        # max_delta = temp_delta_time2.max()
        # mean_delta = temp_delta_time2.mean()
        # std_delta = temp_delta_time2.std()
        # print("temp_delta_time min: {}, max: {}, mean: {}, std: {}".format(min_delta, max_delta, mean_delta, std_delta))

        if NORM_METHOD == 'log':
            # 这里我们使用对数函数来对间隔时间进行缩放
            # + 1.0 + 1e-6  保证结果为正数
            delta_time = np.log(np.array(delta_time) + 1.0 + 1e-6)  # log不写底数时默认以e为底
        elif NORM_METHOD == 'mm':
            temp_delta_time = np.array(delta_time)
            min_delta = temp_delta_time.min()
            max_delta = temp_delta_time.max()
            # (temp_delta_time - min_time) / (max_time - min_time)
            delta_time = (np.array(delta_time) - min_delta) / (max_delta - min_delta)
        elif NORM_METHOD == 'hour':
            delta_time = np.array(delta_time) / 3600

        time_accumulate = [0]
        for delta in delta_time[:-1]:
            next_time = time_accumulate[-1] + delta
            time_accumulate.append(next_time)

        # 过滤-1的歌曲
        new_item_seq = []
        new_time_accumulate = []
        valid_count = 0
        for i in range(len(item_seq)):  # 过滤掉 -1 的item
            if item_seq[i] != -1:
                new_item_seq.append(item_seq[i])
                new_time_accumulate.append(time_accumulate[i])
                valid_count += 1
            if valid_count >= max_length:
                break

        if len(new_item_seq) < min_length:  # 跳过过滤之后的交互记录少于min_length的用户
            continue
        else:
            valid_user_count += 1
            user_index = item2index['user_' + user_id]
            # baseline的用户index从0开始
            user_index_baseline = user_index - top_n_item
            # 在 hash() 对对象使用时，所得的结果不仅和对象的内容有关，还和对象的 id()，也就是内存地址有关。
            index_hash_remaining = user_index % 10
            if index_hash_remaining < 2:  # 20%用户的前50%序列训练，后50%作为测试 （next item 和 next new item）；
                half_index = int(len(new_item_seq) / 2)
                for i in range(half_index):  # 前50%序列，训练
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\n')
                for i in range(half_index, int(len(new_item_seq))):  # 后50%序列，测试
                    out_te_old_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\n')
                    out_te_new_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\n')

                # baseline
                out_tr_baseline_uit.write(str(user_index_baseline) + ',')
                out_tr_baseline_uit.write(':'.join(str(x) for x in new_item_seq[:half_index]) + '\n')

                out_te_baseline_old_uit.write(str(user_index_baseline) + ',')
                out_te_baseline_old_uit.write(':'.join(str(x) for x in new_item_seq[half_index:]) + '\n')

                out_te_baseline_new_uit.write(str(user_index_baseline) + ',')
                out_te_baseline_new_uit.write(':'.join(str(x) for x in new_item_seq[half_index:]) + '\n')

                # lstm baseline
                out_tr_lstm_uit.write(str(user_index_baseline) + ',')
                out_tr_lstm_uit.write(':'.join(str(x) for x in new_item_seq[:half_index]) + '\n')

                out_te_lstm_old_uit.write(str(user_index_baseline) + ',')
                out_te_lstm_old_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

                out_te_lstm_new_uit.write(str(user_index_baseline) + ',')
                out_te_lstm_new_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

            else:  # 80%用户的所有序列作为训练
                for i in range(len(new_item_seq)):
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\n')

                # baseline
                out_tr_baseline_uit.write(str(user_index_baseline) + ',')
                out_tr_baseline_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')
                # lstm baseline
                out_tr_lstm_uit.write(str(user_index_baseline) + ',')
                out_tr_lstm_uit.write(':'.join(str(x) for x in new_item_seq) + '\n')

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
    BASE_DIR = './datasets/'
    DATA_SOURCE = 'lastfm-dataset-1K'
    NORM_METHOD = 'hour'
    # top_n_item = 5000  # 频率排前topn的音乐
    # top_n_item = 10000  # 频率排前topn的音乐
    top_n_item_list = [10000, 15000, 20000, 25000, 30000]  # 频率排前topn的音乐

    user_count = 992

    min_length = 100  # more than
    # max_length = 200  # more than
    max_length = 1000  # more than
    # top_n_user = 1000  # 频率排前topn的user
    # top_n_user = 900  # 频率排前topn的user
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
    #                    names=['userid', 'timestamp', 'artid', 'artname', 'traid', 'tranname'])
    # quotechar=None, quoting=3 避免导致错误的字符 "
    # data = pd.read_csv(path, sep='\t',
    #                    error_bad_lines=False,
    #                    header=None,
    #                    names=['userid', 'timestamp', 'artid', 'artname', 'traid', 'tranname'],
    #                    quotechar=None, quoting=3)

    data = pd.read_csv(
        path, sep='\t', header=None, # nrows=20, #限定读取行数
        names=['userid', 'timestamp', 'artid', 'artname', 'traid', 'tranname'],
        skiprows=[
            2120260 - 1, 2446318 - 1, 11141081 - 1,
            11152099 - 1, 11152402 - 1, 11882087 - 1,
            12902539 - 1, 12935044 - 1, 17589539 - 1
        ],
        quotechar=None, quoting=3
    )
    data = data.dropna(axis=0, how='any')
    print("finish reading csv")
    # 避免重名的歌曲和歌手
    data['tran_name_id'] = data['tranname'] + data['traid']
    data['art_name_id'] = data['artname'] + data['artid']
    for top_n_item in top_n_item_list:
        print("starting processing for top_n_item = {}".format(top_n_item))
        generate_data(user_count, top_n_item, min_length, max_length, data, BASE_DIR, DATA_SOURCE, small_data=False)

    # generate_data()
