import scipy.io as scio
import os
import copy

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch as t
from torch.autograd import Variable
import torchvision.transforms as transforms
import random


# np.set_printoptions(threshold=1e6)

# class MaskProcessor :
#
#     def __init__(self, file_path):
#         self.fileList = os.listdir(file_path)
#
#
#
#     def load_mask(self,size):
#         for i in range(len(self.filelist)):
#             self.mask=self.fileList[i].split("_")
#             first = int(self.mat[2])
#             last = int(self.mat[3].split(".")[0])
#             file = "RECT_ARRAY_%i_%i.mat" % (first, last)
#             data_dir = os.path.join(self.file_path, file)
#             mask_data = scio.loadmat(data_dir)  # 读出来是个字典
#             data = mask_data['mask']
#             numpy_mask = np.transpose(data)


class LoadDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None, xy=0, mode=None, split_n=0.8, rng_seed=50):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为csv
        """
        self.mode = mode
        self.split_n = split_n
        self.rng_seed = rng_seed
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.mask_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.masks = self.read_file(self.mask_path)
        self.labels = self.read_file(self.label_path)

        c = list(zip(self.masks, self.labels))
        random.seed(self.rng_seed)
        random.shuffle(c)
        self.masks, self.labels = zip(*c)

        split_idx = int(len(self.masks) * self.split_n)
        # split_idx = int(100 * self.split_n)
        if self.mode == "train":
            self.masks = self.masks[:split_idx]  # 数据集90%训练
            self.labels = self.labels[:split_idx]
        elif self.mode == "valid":
            self.masks = self.masks[split_idx:]
            self.labels = self.labels[split_idx:]

        # 3 初始化数据处理函数设置
        self.crop_size = crop_size
        self.xy = xy

    def __getitem__(self, index):
        mask = self.masks[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        # img = Image.open(img)
        mask = self.read_mask(mask)
        # print(type(mask[0][0]))

        f = mask.shape[0]
        l = mask.shape[1]
        mask = np.rot90(mask, 1)
        label = self.read_csv(label, f, l, xy=self.xy)

        # label = Image.open(label).convert('RGB')
        #
        # img, label = self.center_crop(img, label, self.crop_size)
        mask = self.center_crop(mask, f, l)

        label = self.center_crop(label, f, l)
        # img, label = self.img_transform(img, label)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        mask = mask[np.newaxis, :]

        label = label[np.newaxis, :]
        # # print(type(mask))
        # print(label)
        # print(label[0][0][0])

        label = label.astype(np.float32)
        # print(label)
        mask = t.from_numpy(mask)
        label = t.from_numpy(label)
        # mask=self.img_transform(mask)
        # label=self.img_transform(label)
        # print(label)
        # print(label[0][0][0])
        # label=label.float()
        # print(label[0][0][0])
        mask = mask.float()
        # print(label)

        sample = {'mask': mask, 'label': label}

        return sample

    def __len__(self):
        return len(self.masks)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def read_mask(self, mask):
        mask_data = scio.loadmat(mask)  # 读出来是个字典
        data = mask_data['mask']
        numpy_mask = np.transpose(data)

        return numpy_mask

    def read_csv(self, label, f, l, xy=0):
        matrix = pd.read_csv(
            label,
            usecols=[0], header=None, names=['index'])
        csv_1 = matrix[matrix['index'] == "Array_2D: %i\t%i" % (f, l)].index.tolist()
        ma = []
        mk = []
        csv_2 = copy.deepcopy(csv_1)
        for i in range(0, len(csv_2)):
            n = csv_2[i] + 1
            m = csv_2[i] + l + 1
            rows = matrix.iloc[n:m, 0]
            mat = np.array(rows)
            for j in range(l):
                row = mat[j].split()
                # print(row)
                a = list(map(float, row))
                a_array = np.array(a)
                mat[j] = a_array

            k = np.stack(mat)
            ma.append(k)
            mk = np.stack(ma, axis=0)
            # print(mk)
        xx_real = mk[0] * np.cos((mk[4] / 180) * np.pi)
        xx_imag = mk[0] * np.sin((mk[4] / 180) * np.pi)
        yy_real = mk[3] * np.cos((mk[7] / 180) * np.pi)
        yy_imag = mk[3] * np.sin((mk[7] / 180) * np.pi)
        if xy == 0:
            label = xx_real
        elif xy == 1:
            label = xx_imag
        elif xy == 2:
            label = yy_real
        else:
            label = yy_imag
        return label

    def center_crop(self, numpy_data, f, l):
        """裁剪输入的图片和标签大小"""
        v = 300
        h = 300
        v = (v // l + 1) if (v // l) % 2 == 0 else (v // l + 2)
        # if (v// l) % 2 == 0:
        #
        #     v = v // l + 1
        # else:
        #     v = v // l + 2
        h = (h // f + 1) if (h // f) % 2 == 0 else (h // f + 2)
        # if (h // f) % 2 == 0:
        #     h = h // f + 1
        # else:
        #     h = h // f + 2
        for i in range(h):
            if i == 0:
                num = numpy_data
            else:
                num = np.hstack((num, numpy_data))
            if i == (h - 1):
                for j in range(v):
                    if j == 0:
                        num_v = num
                    else:
                        num_v = np.vstack((num_v, num))
        num_r = num_v.shape[0]
        num_v1 = num_v.shape[1]

        r1 = int(num_r / 2 - 150)
        r2 = int(num_r / 2 + 150)
        v1 = int(num_v1 / 2 - 150)
        v2 = int(num_v1 / 2 + 150)
        num_v = num_v[r1:r2, v1:v2]
        return num_v

    def img_transform(self, img):
        """对图片和标签做一些数值处理"""

        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485], [0.229])
            ]
        )
        img = transform_img(img)

        return img


#
#
# label_processor = LabelProcessor(cfg.class_dict_path)
device = t.device('cpu')
BASE_DIR = r'E:\Datasets\近场数据\RECT_ARRAY_MASK'
BASE_DIR1 = r"E:\Datasets\近场数据\RECT_ARRAY_NF_x0_y0"
if __name__ == "__main__":
    train = LoadDataset([BASE_DIR, BASE_DIR1], (300, 300), mode='train', split_n=0.8)
    train_data = DataLoader(train, batch_size=20, shuffle=True, num_workers=1)
    for i, sample in enumerate(train_data):
        # 载入数据
        img_data = Variable(sample['label'].to(device))
        print(img_data.shape)

# print("1")
