import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import torch as t
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt


class LoadDataset(Dataset):
    def __init__(self, root,  mode="train"):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为csv
        """
        # 获取该文件夹所有符合模式匹配格式的文件，变成list返回
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        # 1.读取mask的数据
        mask = np.load(self.files_A[index % len(self.files_A)])
        # label是用来验证反向生成的
        mask_label = t.from_numpy(mask).long()
        # source 是用来正向生成的
        mask_source = t.from_numpy(mask[np.newaxis, :, :]).float()

        # 2.读取近场数据
        near_field = np.load(self.files_B[index % len(self.files_B)])

        # 3.读取近场数据
        xx_real = near_field["xx_real"][np.newaxis, :, :]
        xx_imag = near_field["xx_imag"][np.newaxis, :, :]
        yy_real = near_field["yy_real"][np.newaxis, :, :]
        yy_imag = near_field["yy_imag"][np.newaxis, :, :]
        xy_real = near_field["xy_real"][np.newaxis, :, :]
        xy_imag = near_field["xy_imag"][np.newaxis, :, :]
        yx_real = near_field["yx_real"][np.newaxis, :, :]
        yx_imag = near_field["yx_imag"][np.newaxis, :, :]

        # 3.1 合并为 8*256*256的输出样式 c*h*w

        nf = np.concatenate((xx_real, xx_imag, yy_real, yy_imag, xy_real, xy_imag, yx_real, yx_imag), axis=0)
        nf = t.from_numpy(nf).float()

        return {"A": mask_source, "B": nf, "C": mask_label}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


if __name__ == '__main__':
    # transformations
    temp_transforms_ = [
        # transforms.Normalize(mean=[0.0062, 0.0048], std=[1.0016, 1.0003])
        transforms.Normalize(mean=[0.024178, 0.011692, 0.026204, 0.013938, 0.00000000000022865, 0.00000000000020783, 0.00000000000011487, -0.000000000000089899],
                             std=[0.12494, 0.11972, 0.13355, 0.13404, 0.00013799, 0.0000403454, 0.000062564, 0.00001534])
    ]

    device = t.device('cpu')
    train = LoadDataset("../datasets/crop_128/final", mode="train")
    train_data = DataLoader(train, batch_size=1, shuffle=True, num_workers=0)

    nb_samples = len(train_data)
    channel_mean = t.zeros(8)
    channel_std = t.zeros(8)


    for i, sample in enumerate(train_data):
        # 载入数据
        nf = Variable(sample['B'].to(device))

        # nf_xx_r = nf[0, 0, :, :].cpu().detach().numpy()
        # nf_xx_i = nf[0, 1, :, :].cpu().detach().numpy()
        # nf_yy_r = nf[0, 2, :, :].cpu().detach().numpy()
        # nf_yy_i = nf[0, 3, :, :].cpu().detach().numpy()
        # nf_xy_r = nf[0, 4, :, :].cpu().detach().numpy()
        # nf_xy_i = nf[0, 5, :, :].cpu().detach().numpy()
        # nf_yx_r = nf[0, 6, :, :].cpu().detach().numpy()
        # nf_yx_i = nf[0, 7, :, :].cpu().detach().numpy()
        #
        # plt.imshow(nf_xy_r)
        # plt.show()
        #
        # plt.imshow(nf_xy_i)
        # plt.show()
        #
        # plt.imshow(nf_yx_r)
        # plt.show()
        #
        # plt.imshow(nf_yx_i)
        # plt.show()
        #
        # break

        N, C, H, W = nf.shape[:4]
        nf = nf.view(N, C, -1)
        channel_mean += nf.mean(2).sum(0)
        channel_std += nf.std(2).sum(0)

    # 获取同一batch的均值和标准差
    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)


    # nb_samples = len(train_data)
    # channel_mean = t.zeros(8)
    # channel_std = t.zeros(8)
    # for i, sample in enumerate(train_data):
    #     # 载入数据
    #
    #     data = Variable(sample['B'].to(device))
    #
    #
    #     N, C, H, W = data.shape[:4]
    #     data = data.view(N, C, -1)
    #     channel_mean += data.mean(2).sum(0)
    #     channel_std += data.std(2).sum(0)
    #
    #
    #
    # #获取同一batch的均值和标准差
    # channel_mean /= nb_samples
    # channel_std /= nb_samples
    # print(channel_mean, channel_std)