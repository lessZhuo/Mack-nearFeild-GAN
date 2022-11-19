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
        mask = mask[:, :, np.newaxis]
        mask = mask.astype(np.float)
        mask = self.transform(mask).float()

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
        nf_real = nf_real[:, :, np.newaxis]
        nf_imag = nf_imag[:, :, np.newaxis]

        # 2.3根据是否合并或者读取实部虚部进行返回数据
        if self.combine:
            nf = np.concatenate((nf_real, nf_imag), axis=2)
            nf = self.transform(nf).float()
            return {"A": mask, "B": nf}
        else:
            if self.part == "real":
                return {"A": mask, "B": self.transform(nf_real).float()}
            else:
                return {"A": mask, "B": self.transform(nf_imag).float()}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


# transformations
transforms_ = [
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize([1], [1]),
]


class MaskNfDatasetV2(Dataset):
    def __init__(self, root, mode="train", combine=False, direction="x", part="real"):
        self.combine = combine
        self.direction = direction
        self.part = part
        # 获取该文件夹所有符合模式匹配格式的文件，变成list返回
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):

        # 1.读取mask的数据
        mask = np.load(self.files_A[index % len(self.files_A)])
        mask_label = t.from_numpy(mask).long()

        mask_zero = mask ^ 1
        mask_one = mask ^ 0

        mask = np.array([mask_zero, mask_one])
        mask = mask.astype(np.float)
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
            nf = np.concatenate((nf_real, nf_imag), axis=2)
            nf = t.from_numpy(nf).float()
            return {"A": mask, "B": nf, "C": mask_label}
        else:
            if self.part == "real":
                return {"A": mask, "B": t.from_numpy(nf_real).float(), "C": mask_label}
            else:
                return {"A": mask, "B": t.from_numpy(nf_imag).float(), "C": mask_label}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


if __name__ == '__main__':
    device = t.device('cpu')
    train = MaskNfDatasetV2("../datasets/crop_256", combine=True, direction="x")
    train_data = DataLoader(train, batch_size=20, shuffle=True, num_workers=0)
    for i, sample in enumerate(train_data):
        # 载入数据
        img_data = Variable(sample['A'].to(device))
        mmm = Variable(sample['B'].to(device))
        label = Variable(sample['C'].to(device).long())

        img_data_ = img_data[0, :, :, :]
        img_data_d = img_data_.max(dim=0)[1].data.squeeze().detach().numpy()
        print(img_data_d.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(img_data_d)
        plt.show()
        plt.close()
        break
