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


class LoadDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", combine=False, direction="x", part="real"):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为csv
        """
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
        nf = t.from_numpy(nf)
        nf = self.transform(nf).float()

        return {"A": mask_source, "B": nf, "C": mask_label}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


if __name__ == '__main__':
    # transformations
    transforms_ = [
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize([1], [1]),
    ]

    device = t.device('cpu')
    train = LoadDataset("../datasets/crop_256/final", transforms_=transforms_, combine=False, direction="x", part="imag")
    train_data = DataLoader(train, batch_size=1, shuffle=True, num_workers=0)
    for i, sample in enumerate(train_data):
        # 载入数据
        img_data = Variable(sample['A'].to(device))
        mmm = Variable(sample['B'].to(device))

        print(mmm)

        break
