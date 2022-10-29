import scipy.io as scio
import os
import copy
import torch

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable


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
        # print(c)

        self.masks, self.labels = zip(*c)

    def __getitem__(self, index):
        mask = self.masks[index]
        # print(mask)
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        # img = Image.open(img)
        mask = self.read_mask(mask)
        print(type(mask[0][0]))

        f = mask.shape[0]
        l = mask.shape[1]

        np.rot90(mask,1)


        label_xx_real = self.read_csv(label, f, l, 0)
        label_xx_imag = self.read_csv(label, f, l, 1)
        label_yy_real = self.read_csv(label, f, l, 2)
        label_yy_imag = self.read_csv(label, f, l, 3)
        print(label_xx_real.shape)
        print(label_xx_imag.shape)
        print(label_yy_real.shape)
        print(label_yy_imag.shape)

        mask = self.center_crop(mask, f, l)

        label_xx_real = self.center_crop(label_xx_real, f, l)
        label_xx_imag = self.center_crop(label_xx_imag, f, l)
        label_yy_real = self.center_crop(label_yy_real, f, l)
        label_yy_imag = self.center_crop(label_yy_imag, f, l)

        print(label_xx_real.shape)
        print(label_xx_imag.shape)
        print(label_yy_real.shape)
        print(label_yy_imag.shape)
        print(mask.shape)
        label = np.array([label_xx_real, label_xx_imag, label_yy_real, label_yy_imag])
        print(label.shape)
        sample = {'mask': mask, 'label': label}
        # print(sample)

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
        numpy_mask = np.transpose(data)  # 转置()T

        return numpy_mask

    def read_csv(self, label, f, l, xy=0):
        matrix = pd.read_csv(label, usecols=[0], header=None, names=['index'])
        print(label)
        csv_1 = matrix[matrix['index'] == "Array_2D: %i\t%i" % (f, l)].index.tolist()
        # print(label, csv_1, f, l)
        ma = []
        mk = []
        csv_2 = copy.deepcopy(csv_1)
        # print(csv_2)
        for i in range(0, len(csv_2)):
            n = csv_2[i] + 1
            m = csv_2[i] + l + 1
            # print(n, m)
            rows = matrix.iloc[n:m, 0]
            mat = np.array(rows)
            # print(rows, mat)
            for j in range(l):
                row = mat[j].split()
                a = list(map(float, row))
                a_array = np.array(a)
                mat[j] = a_array
                # print(mat[j])

            k = np.stack(mat)
            # print(k)
            ma.append(k)
            mk = np.stack(ma, axis=0)
            # print(mk.shape)

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

        h = (h // f + 1) if (h // f) % 2 == 0 else (h // f + 2)

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
                        # print(num_v.shape)
        num_r = num_v.shape[0]
        num_v1 = num_v.shape[1]
        # print(num_r, num_v1)

        r1 = int(num_r / 2 - 256 / 2)
        r2 = int(num_r / 2 + 256 / 2)
        v1 = int(num_v1 / 2 - 256 / 2)
        v2 = int(num_v1 / 2 + 256 / 2)
        num_v = num_v[r1:r2, v1:v2]
        return num_v


def read_file(path):
    """从文件夹中读取数据"""
    files_list = os.listdir(path)
    file_path_list = [os.path.join(path, img) for img in files_list]
    file_path_list.sort()
    return file_path_list


def read_mask(mask):
    mask_data = scio.loadmat(mask)  # 读出来是个字典
    data = mask_data['mask']
    # numpy_mask = np.transpose(data)  # 转置()T

    return data


def read_csv(label, f, l, xy=0):
    matrix = pd.read_csv(label, usecols=[0], header=None, names=['index'])

    csv_1 = matrix[matrix['index'] == "Array_2D: %i\t%i" % (f, l)].index.tolist()
    # print(label, csv_1, f, l)
    ma = []
    mk = []
    csv_2 = copy.deepcopy(csv_1)
    # print(csv_2)
    for i in range(0, len(csv_2)):
        n = csv_2[i] + 1
        m = csv_2[i] + l + 1
        # print(n, m)
        rows = matrix.iloc[n:m, 0]
        mat = np.array(rows)
        # print(rows, mat)
        for j in range(l):
            row = mat[j].split()
            a = list(map(float, row))
            a_array = np.array(a)
            mat[j] = a_array
            # print(mat[j])

        k = np.stack(mat)
        # print(k)
        ma.append(k)
        mk = np.stack(ma, axis=0)
        # print(mk.shape)

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


def center_crop(numpy_data, f, l):
    """裁剪输入的图片和标签大小"""
    v = 300
    h = 300

    v = (v // l + 1) if (v // l) % 2 == 0 else (v // l + 2)

    h = (h // f + 1) if (h // f) % 2 == 0 else (h // f + 2)

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
                    # print(num_v.shape)
    num_r = num_v.shape[0]
    num_v1 = num_v.shape[1]
    # print(num_r, num_v1)

    r1 = int(num_r / 2 - 256 / 2)
    r2 = int(num_r / 2 + 256 / 2)
    v1 = int(num_v1 / 2 - 256 / 2)
    v2 = int(num_v1 / 2 + 256 / 2)
    num_v = num_v[r1:r2, v1:v2]
    return num_v


def corp(data, len):
    # 1.获取数据的长度和宽度
    rowLength = data.shape[0]
    colLength = data.shape[1]
    # 2.根据要拆分的数据进行切分
    # 分为5部分，左上，右上，左下，右下
    data_left_up = data[:len,:len]
    data_left_down = data[len:, :len]
    data_right_up = data[:len, len:]
    data_right_down = data[len:, :len:]
    data_mid = data[rowLength/2-len/2:rowLength/2+len/2,colLength/2-len/2:colLength/2+len/2]

    return data_left_up,data_left_down,data_right_up,data_right_down,data_mid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = r'E:\QQfile\近场数据\RECT_ARRAY_MASK'
BASE_DIR1 = r"E:\QQfile\近场数据\RECT_ARRAY_NF_x0_y0"
if __name__ == "__main__":
    masks_path = read_file(BASE_DIR)
    labels_path = read_file(BASE_DIR1)

    c = list(zip(masks_path, labels_path))
    # print(c)

    masks, labels = zip(*c)

    for mask, label in c:
        mask = read_mask(mask)

        f = mask.shape[1]
        l = mask.shape[0]
        print(mask.shape)
        label_xx_real = read_csv(label, f, l, 0)
        label_xx_imag = read_csv(label, f, l, 1)
        label_yy_real = read_csv(label, f, l, 2)
        label_yy_imag = read_csv(label, f, l, 3)
        print(label_xx_real.shape)
        mask = center_crop(mask, f, l)

        label_xx_real = center_crop(label_xx_real, f, l)
        label_xx_imag = center_crop(label_xx_imag, f, l)
        label_yy_real = center_crop(label_yy_real, f, l)
        label_yy_imag = center_crop(label_yy_imag, f, l)

        # 256*256


        print(mask.shape)
        label = np.array([label_xx_real, label_xx_imag, label_yy_real, label_yy_imag])
        print(label.shape)

    # train = LoadDataset([BASE_DIR, BASE_DIR1], (256, 256), mode='train', split_n=0.8)
    # train_data = DataLoader(train, batch_size=20, shuffle=True, num_workers=0)
    # # print(train)
    # for i, sample in enumerate(train_data):
    #     # print(i, sample)
    #     # 载入数据
    #     img_data = Variable(sample['label'].to(device))
    #     print(img_data.shape)
