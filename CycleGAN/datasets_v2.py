import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch as t
import matplotlib.pyplot as plt


class MaskNfDatasetV2(Dataset):
    def __init__(self, root, transforms_=None, mode="train", combine=False, direction="x", part="real"):
        self.combine = combine
        self.direction = direction
        self.part = part
        self.transform = transforms.Compose(transforms_)
        # 获取该文件夹所有符合模式匹配格式的文件，变成list返回
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        # 1.读取mask的数据
        mask = np.load(self.files_A[index % len(self.files_A)])
        mask_temp = t.from_numpy(mask[np.newaxis, :, :]).float()
        mask_label = t.from_numpy(mask).long()

        mask_zero = mask ^ 1
        mask_one = mask ^ 0

        mask = np.array([mask_zero, mask_one])
        mask = mask.astype(float)
        mask = t.from_numpy(mask)

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
        nf = t.from_numpy(nf)
        nf = self.transform(nf).float()

        return {"A": mask, "B": nf, "C": mask_label, "D": mask_temp}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


if __name__ == '__main__':

    transforms_ = [
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize([1], [1]),
    ]

    device = t.device('cpu')
    train = MaskNfDatasetV2("../datasets/crop_256/new", transforms_=transforms_, combine=True, direction="x")
    train_data = DataLoader(train, batch_size=2, shuffle=True, num_workers=0)

    # tf = transforms.Compose(de_transforms_)
    # # 计算原图的 mean 和std
    # nb_samples = 0
    # # 创建3维的空列表
    # channel_mean = t.zeros(2)
    # channel_std = t.zeros(2)

    for i, sample in enumerate(train_data):
        # 载入数据
        img_data = Variable(sample['A'].to(device))
        mmm = Variable(sample['B'].to(device))
        # label = Variable(sample['C'].to(device).long())
        # print(mmm)
        # mm = tf(mmm)
        B_r = mmm[0, 0, :, :]
        B_i = mmm[0, 1, :, :]
        B_r = B_r.cpu().squeeze().detach().numpy()
        B_i = B_i.cpu().squeeze().detach().numpy()

        plt.imshow(B_r)
        plt.colorbar()
        plt.show()

        plt.imshow(B_i)
        plt.colorbar()
        plt.show()

        break
    #     # print(mmm.shape)
    #     N, C, H, W = mmm.shape[:4]
    #     # 将w,h维度的数据展平，为batch，channel,data,然后对三个维度上的数分别求和和标准差
    #     image = mmm.view(N, C, -1)
    #     # print(image.shape)
    #     # 展平后，w,h属于第二维度，对他们求平均，sum(0)为将同一纬度的数据累加
    #     channel_mean += image.mean(2).sum(0)
    #     # 展平后，w,h属于第二维度，对他们求标准差，sum(0)为将同一纬度的数据累加
    #     channel_std += image.std(2).sum(0)
    #     # 获取所有batch的数据，这里为1
    #     nb_samples += N
    #     # 获取同一batch的均值和标准差
    #
    # channel_mean /= nb_samples
    # channel_std /= nb_samples
    # print(channel_mean, channel_std)
    #
