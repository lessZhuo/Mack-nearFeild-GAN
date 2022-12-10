import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch as t
import matplotlib.pyplot as plt


class MaskNfDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", combine=False, direction="x", part="real"):
        self.transform = transforms.Compose(transforms_)
        self.combine = combine
        self.direction = direction
        self.part = part
        # 获取该文件夹所有符合模式匹配格式的文件，变成list返回
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):

        # 1.读取mask的数据
        mask = np.load(self.files_A[index % len(self.files_A)])
        # mask = mask[:, :, np.newaxis]  #---
        # mask = mask.astype(np.float)  #---
        mask = mask[np.newaxis, :, :]  # ---
        mask = mask.astype(np.float)  # ---
        mask = t.from_numpy(mask).float()  # ---

        # 2.读取近场数据
        near_field = np.load(self.files_B[index % len(self.files_B)])

        # 2.1判读读取xx方向还是yy方向数据
        if self.direction == "x":
            nf_real = near_field["xx_real"]
            nf_imag = near_field["xx_imag"]
        else:
            nf_real = near_field["yy_real"]
            nf_imag = near_field["yy_imag"]

        # 2.2进行类型转换和维度扩充
        nf_real = nf_real.astype(np.float32)
        nf_imag = nf_imag.astype(np.float32)
        # nf_real = nf_real[:, :, np.newaxis] ---
        # nf_imag = nf_imag[:, :, np.newaxis] ---
        nf_real = nf_real[np.newaxis, :, :]  # ---
        nf_imag = nf_imag[np.newaxis, :, :]  # ---

        # 2.3根据是否合并或者读取实部虚部进行返回数据
        if self.combine:
            nf = np.concatenate((nf_real, nf_imag), axis=0)  # ---
            nf = t.from_numpy(nf)  # ---
            nf = self.transform(nf).float()
            return {"A": mask, "B": nf}
        else:
            if self.part == "real":
                return {"A": mask, "B": self.transform(t.from_numpy(nf_real)).float()}  # ---
            else:
                return {"A": mask, "B": self.transform(t.from_numpy(nf_imag)).float()}  # ---

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


# transformations
transforms_ = [
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize(mean=[0.193, 0.195], std=[0.927, 1.378]),
]

de_transforms_ = [
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize(mean=[-0.2031, -0.1365], std=[11.0375, 7.4184]),
]


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

        # 2.1判读读取xx方向还是yy方向数据
        if self.direction == "x":
            nf_real = near_field["xx_real"]
            nf_imag = near_field["xx_imag"]
        else:
            nf_real = near_field["yy_real"]
            nf_imag = near_field["yy_imag"]

        # 2.2进行类型转换和维度扩充
        nf_real = nf_real.astype(np.float32)
        nf_imag = nf_imag.astype(np.float32)
        nf_real = nf_real[np.newaxis, :, :]
        nf_imag = nf_imag[np.newaxis, :, :]

        # 2.3根据是否合并或者读取实部虚部进行返回数据
        if self.combine:
            nf = np.concatenate((nf_real, nf_imag), axis=0)
            nf = t.from_numpy(nf).float()
            nf = self.transform(nf)
            return {"A": mask, "B": nf, "C": mask_label, "D": mask_temp}
        else:
            if self.part == "real":
                return {"A": mask, "B": self.transform(t.from_numpy(nf_real).float()), "C": mask_label, "D": mask_temp}
            else:
                return {"A": mask, "B": self.transform(t.from_numpy(nf_imag).float()), "C": mask_label, "D": mask_temp}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


if __name__ == '__main__':
    device = t.device('cpu')
    train = MaskNfDataset("../datasets/crop_256/new", transforms_=transforms_, combine=True, direction="x")
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
