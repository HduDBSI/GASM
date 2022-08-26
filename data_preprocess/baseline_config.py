# -*- coding: utf-8 -*-

# common
min_length = 100  # more than
max_length = 1500  # less than
min_length_lfm1b = 1000  # more than
max_length_lfm1b = 1500  # less than
# dataset = ['last_music', '30music', 'xiami_music']
user_count_list = [900, 3000, 4000, 8000]

# stamp_cuda = "0"
hrm_avg_cuda = "0"
hrm_max_cuda = "1"
shan_cuda = "2"
tisasrec_cuda = "3"

# =======================================
# lastfm
top_n_item_dict_lastfm = {50: 66407, 100: 31557, 150: 19317, 200: 13539, 250: 9985}
# listP_lastfm = [50, 100, 150, 200, 250, 300]
freq_list_lastfm = [50]
# freq_list_lastfm = [250]

# =======================================
# lfm1b
# top_n_item_dict_lfm1b = {0:11, 250: 116139}
top_n_item_dict_lfm1b = {0: 11, 250: 116139, 280: 102416}
# freq_list_lfm1b = [250, 280]
freq_list_lfm1b = [280]

# =======================================
# 30music
#top_n_item_dict_30music = {50: 84882, 100: 38422, 150: 23315, 200: 15898, 250: 11807}
top_n_item_dict_30music = {50: 90868, 100: 38422, 150: 23315, 200: 15898, 250: 11807}
freq_list_30music = [50]


# =======================================
# xiami
top_n_item_dict_xiami = {10: 64334, 20: 35395, 30: 24420, 40: 18629, 50: 14961}
# freq_list_xiami = [10, 20, 30, 40, 50]
freq_list_xiami = [10]

top_n_item_dict = [top_n_item_dict_lastfm, top_n_item_dict_30music, top_n_item_dict_xiami]
freq_list = [freq_list_lastfm, freq_list_30music, freq_list_xiami, freq_list_lfm1b]