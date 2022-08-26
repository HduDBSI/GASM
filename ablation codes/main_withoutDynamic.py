from model_withoutDynamic import *
from utils_withoutDynamic import *

SEED = 42
sys.setrecursionlimit(100000)  # 设置递归深度
torch.manual_seed(1000)  # 控制Embedding初始化值
torch.backends.cudnn.deterministic = True  # 设置cuda随机种子

parser = argparse.ArgumentParser()

parser.add_argument('--cuda_device', default='0', help='0/1/2/3')
parser.add_argument('--output', default='display', help='local/display')
parser.add_argument('--dataset', default='last',
                    help='dataset name: last/30Music/xiami')

parser.add_argument('--detail', default='GASM_alpha', help='description of code to distinguish')
parser.add_argument('--valid_portion', default=8, type=int, help='the portion of data-partition(train-test):1~9/0')
parser.add_argument('--drop_portion', default=0, type=float, help='the portion of remain item:0.3/0.4/0.5')
parser.add_argument('--data_size', default=1, type=float, help='part of data')
parser.add_argument('--slide_step', default=1, type=int,
                    help='the step of slide window: if slide_step > 5,there is no overlap')
parser.add_argument('--windowLenth', type=int,
                    default=5, help='maxlenth of slide window')

parser.add_argument('--batchSize', type=int,
                    default=256, help='input batch size')
parser.add_argument('--hiddenSize', type=int,
                    default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=25,
                    help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1,
                    help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3,
                    help='the number of steps after which the learning rate decay')
# [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true',
                    help='only use the global preference to predict')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device


def main():
    print("model:{}".format('GASM'))
    start = time.time()
    day = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if opt.output == 'local':
        file = open('./result/' + opt.dataset + day + '.txt', 'w+')
        __stderr__ = sys.stderr
        sys.stderr = file  # 输出报错信息
        sys.stdout = file

    print(opt)
    print(day)

    type = ['last', '30Music', 'xiami']
    for data_index in [0, 1, 2]:
    #for data_index in [2]:#lastfm
        opt.dataset = type[data_index]
        print('============= dataset:{} ============='.format(opt.dataset))
        if opt.dataset == 'last':
            n_music = 66407
            n_artist = 7392
            n_album = 9008  # 出现24759次
            n_usr = 900
        elif opt.dataset == 'xiami':
            n_music = 64334
            n_artist = 16243
            n_album = 32696
            n_usr = 4000
        elif opt.dataset == '30Music':
            n_music = 90868
            n_artist = 11829
            n_album = 18333
            n_usr = 3000

        test_data_file_path = os.path.join('./' + opt.dataset + '_splited_test')
        train_data_file_path = os.path.join('./' + opt.dataset + '_splited_train')

        print("---------------data input---------------")

        train_data, test_data, test_new_data = data_input(opt)

        print("Total split train records: {}".format(len(train_data[0])))
        print("Total split test records (next-one):{}".format(len(test_data[0])))
        print("Total split test records (next-new):{}".format(len(test_new_data[0])))
        #continue
        test_data = Data(test_data, opt)
        train_data = Data(train_data, opt)
        test_new_data = Data(test_new_data,opt)

        model = trans_to_cuda(SessionGraph(opt, n_usr, n_album, n_artist, n_music))

        for epoch in range(opt.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            train_test(model, train_data, test_data, test_new_data)
            sys.stdout.flush()  # 每个epoch直接输出缓存区内容，避免长期没有输出文件
        print('-------------------------------------------------------')

        end = time.time()
        print("Run time: %f s" % (end - start))
        del model #释放对象

if __name__ == '__main__':
    main()
