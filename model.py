import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from collections import defaultdict


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        # 隐藏层个数超参数
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size

        # 需要学习的Parameter
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        # 学习bias偏差
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        # matual 矩阵乘法计算
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # cat 拼接
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_usr, n_album, n_artist, n_music):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_usr = n_usr
        self.n_music = n_music
        self.n_item = n_album + n_artist + n_music + 1  # 多一位0用于padding
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        # 将用户和item的隐藏层分别存储
        self.item_embedding = nn.Embedding(self.n_item, self.hidden_size)
        self.usr_embedding = nn.Embedding(self.n_usr, self.hidden_size)

        self.gnn = GNN(self.hidden_size, step=opt.step)

        self.linear_one_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.linear_one_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.linear_one_3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two_3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.linear_layer2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_transform = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=True)
        self.linear_transform1 = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)
        self.linear_transform2 = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)
        self.linear_transform3 = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)
        self.linear_transform4 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        # self.linear_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, usr_embedding, music_embedding, artist_embedding, album_embedding):  # 传入seq_hidden

        # alpha
        ht = usr_embedding.view(usr_embedding.shape[0], usr_embedding.shape[1], 1)  # batch_size x latent_size

        last_music_embedding = music_embedding[:, -1, :]  # 滑动窗口最右侧的一首歌 batch_size x hidden
        last_artist_embedding = artist_embedding[:, -1, :]
        last_album_embedding = album_embedding[:, -1, :]

        q1_1 = self.linear_one_1(last_music_embedding).view(last_music_embedding.shape[0], 1,
                                                            last_music_embedding.shape[1])
        # batch_size x seq_length x latent_size
        q1_2 = self.linear_two_1(music_embedding)

        q2_1 = self.linear_one_2(last_artist_embedding).view(last_artist_embedding.shape[0], 1,
                                                             last_artist_embedding.shape[1])
        q2_2 = self.linear_two_2(artist_embedding)

        q3_1 = self.linear_one_3(last_album_embedding).view(last_album_embedding.shape[0], 1,
                                                            last_album_embedding.shape[1])
        q3_2 = self.linear_two_3(album_embedding)

        # 激活函数tanh
        alpha = torch.bmm((torch.tanh(q1_1 + q1_2)), ht)  # batch x node x 1
        beta = torch.bmm((torch.tanh(q2_1 + q2_2)), ht)
        delta = torch.bmm((torch.tanh(q3_1 + q3_2)), ht)

        alpha = torch.softmax(alpha, dim=1)
        beta = torch.softmax(beta, dim=1)
        delta = torch.softmax(delta, dim=1)
        a = torch.sum(alpha * music_embedding, 1)  # batch x hidden
        b = torch.sum(beta * artist_embedding, 1)
        c = torch.sum(delta * album_embedding, 1)

        # a，b向量进行layernorm
        layernorm = trans_to_cuda(nn.LayerNorm(a.shape[1], eps=1e-6))
        a = layernorm(a)
        b = layernorm(b)
        c = layernorm(c)

        if not self.nonhybrid:
            st = self.linear_transform1(torch.cat([a, b, c], 1))  # short-term
            dm = self.linear_transform2(
                torch.cat([last_music_embedding, last_artist_embedding, last_album_embedding], 1))  # dynamic
            all = self.linear_transform3(torch.cat([usr_embedding, st, dm], 1))

        item_embedding = self.item_embedding.weight[1:self.n_music + 1]  # n_nodes x latent_size
        scores = torch.matmul(all, item_embedding.transpose(1, 0))

        return scores  # batch_size x item_num

        # beta2：在short-中融入最后一个item；拼接时不考虑使用dynamic
        # ht = usr_embedding.view(usr_embedding.shape[0], usr_embedding.shape[1], 1)  # batch_size x latent_size
        #
        # last_music_embedding = music_embedding[:, -1, :]  # 滑动窗口最右侧的一首歌 batch_size x hidden
        # last_artist_embedding = artist_embedding[:, -1, :]
        # last_album_embedding = album_embedding[:, -1, :]
        #
        # q1_1 = self.linear_one_1(last_music_embedding).view(last_music_embedding.shape[0], 1,
        #                                                     last_music_embedding.shape[1])
        # # batch_size x seq_length x latent_size
        # q1_2 = self.linear_two_1(music_embedding)
        #
        # q2_1 = self.linear_one_2(last_artist_embedding).view(last_artist_embedding.shape[0], 1,
        #                                                      last_artist_embedding.shape[1])
        # q2_2 = self.linear_two_2(artist_embedding)
        #
        # q3_1 = self.linear_one_3(last_album_embedding).view(last_album_embedding.shape[0], 1,
        #                                                     last_album_embedding.shape[1])
        # q3_2 = self.linear_two_3(album_embedding)
        #
        # # 激活函数tanh
        # alpha = torch.bmm((torch.tanh(q1_1 + q1_2)), ht)  # batch x node x 1
        # beta = torch.bmm((torch.tanh(q2_1 + q2_2)), ht)
        # delta = torch.bmm((torch.tanh(q3_1 + q3_2)), ht)
        #
        # alpha = torch.softmax(alpha, dim=1)
        # beta = torch.softmax(beta, dim=1)
        # delta = torch.softmax(delta, dim=1)
        # a = torch.sum(alpha * music_embedding, 1)  # batch x hidden
        # b = torch.sum(beta * artist_embedding, 1)
        # c = torch.sum(delta * album_embedding, 1)
        #
        # # a，b向量进行layernorm
        # layernorm = trans_to_cuda(nn.LayerNorm(a.shape[1], eps=1e-6))
        # a = layernorm(a)
        # b = layernorm(b)
        # c = layernorm(c)
        #
        # st = self.linear_transform1(torch.cat([a, b, c], 1))  # short-term
        # all = self.linear_transform4(torch.cat([usr_embedding, st], 1))
        # item_embedding = self.item_embedding.weight[1:self.n_music + 1]  # n_nodes x latent_size
        # scores = torch.matmul(all, item_embedding.transpose(1, 0))
        #
        # return scores  # batch_size x item_num

    def forward(self, item, usr, A):
        # usr batch x 1
        # item: batch x max_node
        h1 = self.item_embedding(item)  # 取出对应的embedding向量
        h2 = self.usr_embedding(usr.view(usr.shape[0], 1))

        # 拼接成batch x n_node x hidden_size
        # usr_embedding在前
        hidden = torch.cat([h2, h1], dim=1)
        hidden = self.gnn(A, hidden)  # 根据转移矩阵更新向量
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):  # 传入Data数据
    A = data.matrix[i]
    usr = data.usr[i]
    items = data.items[i]  # 一张图中的所有节点id列表

    music_alias_input = data.music_alias_input[i]
    artist_alias_input = data.artist_alias_input[i]
    artist_mask_input=data.artist_mask_input[i]
    album_alias_input = data.album_alias_input[i]
    album_mask_input = data.album_mask_input[i]

    items = trans_to_cuda(torch.Tensor(items).long())
    usr = trans_to_cuda(torch.Tensor(usr).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    album_mask_input = trans_to_cuda(torch.Tensor(album_mask_input).long())
    artist_mask_input = trans_to_cuda(torch.Tensor(artist_mask_input).long())

    hidden = model(items, usr, A)  # 获取GNN的节点特征表示
    # 构建隐藏层推荐序列：usr+music_seq
    usr_embedding = hidden[:, 0, :]
    # usr_embedding=usr_embedding.view(usr_embedding.shape[0],1,usr_embedding.shape[1])
    # get = lambda i: hidden[i][music_alias_input[i]]
    music_embedding = torch.stack(
        [hidden[i][music_alias_input[i]] for i in torch.arange(len(music_alias_input)).long()])
    artist_embedding = torch.stack(
        [hidden[i][artist_alias_input[i]] for i in torch.arange(len(artist_alias_input)).long()])
    album_embedding = torch.stack(
        [hidden[i][album_alias_input[i]] for i in torch.arange(len(album_alias_input)).long()])
    album_embedding = album_embedding * album_mask_input.view(album_mask_input.shape[0], -1, 1)
    artist_embedding = artist_embedding * artist_mask_input.view(artist_mask_input.shape[0], -1, 1)

    targets = data.target[i]
    return targets, model.compute_scores(usr_embedding, music_embedding, artist_embedding, album_embedding)

def predict(model,test_data):
    N = 21
    hit_res = np.zeros(N)
    mrr_res = np.zeros(N)
    pre_res = np.zeros(N)

    model.eval()
    hit = defaultdict(list)
    mrr = defaultdict(list)
    pre = defaultdict(list)
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        # 一个batch的数据
        targets, scores = forward(model, i, test_data)  # score是[0~n_music)
        sub_scores = scores.topk(N - 1)[1]  # 取排名前n,[0,19]，返回score中的元素下标
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()

        # todo:根据推荐分数计算artist， 再取出对应的artist的歌曲分数，建立artist-music字典；在CPU上运行
        for score, target in zip(sub_scores, targets):  # 取出每个batch对应的数据
            target = target - 1  # 对应scores中的下标需要-1
            for topN in range(1, N):
                topN_item = score[:topN]  # 推荐的前n个item [1,20]
                hit[topN].append(np.isin(target, topN_item))
                common = []  # 预测结果和真实结果共有的
                if target in topN_item:
                    common.append(target)
                    mrr[topN].append(1 / (np.where(topN_item == target)[0][0] + 1))
                else:
                    mrr[topN].append(0)

                pre[topN].append(len(common) / len(topN_item))
    # 累计所有batch的数据后进行平均
    for topN in range(1, N):
        hit_res[topN] = np.mean(hit[topN]) * 100
        mrr_res[topN] = np.mean(mrr[topN]) * 100
        pre_res[topN] = np.mean(pre[topN]) * 100
    print("hit:\n{}".format(hit_res[1:]))
    print("mrr:\n{}".format(mrr_res[1:]))
    print("pre:\n{}".format(pre_res[1:]))

def train_test(model, train_data, test_data, test_new_data):
    print('start training: ', datetime.datetime.now())
    model.scheduler.step()
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):  # i遍历图的下标进行访问，j表示循环的次数
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets-1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting (next-one): ', datetime.datetime.now())
    predict(model,test_data)

    print('start predicting (next-new): ', datetime.datetime.now())
    predict(model,test_new_data)