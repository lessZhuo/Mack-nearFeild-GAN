import scipy.io as scio
import os
import copy
import torch

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset

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
    print(rowLength)
    print(colLength)
    # 2.根据要拆分的数据进行切分
    # 分为5部分，左上，右上，左下，右下
    data_left_up = data[:len, :len]
    data_left_down = data[len:, :len]
    data_right_up = data[:len, len:]
    data_right_down = data[len:, :len:]
    data_mid = data[int(rowLength / 2 - len / 2):int(rowLength / 2 + len / 2),
               int(colLength / 2 - len / 2):int(colLength / 2 + len / 2)]

    return data_left_up, data_left_down, data_right_up, data_right_down, data_mid


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
        print("-----------------分割线---------------------------")
        mask = read_mask(mask)

        f = mask.shape[1]
        l = mask.shape[0]
        # print(mask.shape)
        label_xx_real = read_csv(label, f, l, 0)
        label_xx_imag = read_csv(label, f, l, 1)
        label_yy_real = read_csv(label, f, l, 2)
        label_yy_imag = read_csv(label, f, l, 3)
        # print(label_xx_real.shape)

        # 256*256

        mask = center_crop(mask, f, l)

        label_xx_real = center_crop(label_xx_real, f, l)
        label_xx_imag = center_crop(label_xx_imag, f, l)
        label_yy_real = center_crop(label_yy_real, f, l)
        label_yy_imag = center_crop(label_yy_imag, f, l)

        # 128*128
        # 切分mask数据为5个128*128的图片
        # 根据位置进行保存

        mask_left_up, mask_left_down, mask_right_up, mask_right_down, mask_mid = corp(mask, 128)
        np.save("D:\dataset\A\RA_Mask_left_up_%i_%i" % (f, l), mask_left_up)
        np.save("D:\dataset\A\RA_Mask_left_down_%i_%i" % (f, l), mask_left_down)
        np.save("D:\dataset\A\RA_Mask_right_up_%i_%i" % (f, l), mask_right_up)
        np.save("D:\dataset\A\RA_Mask_right_down_%i_%i" % (f, l), mask_right_down)
        np.save("D:\dataset\A\RA_Mask_mask_mid_%i_%i" % (f, l), mask_mid)

        # 切分各种类型的label
        # 根据位置进行保存

        label_xx_real_left_up, label_xx_real_left_down, label_xx_real_right_up, label_xx_real_right_down, label_xx_real_mid = corp(
            label_xx_real, 128)
        label_xx_imag_left_up, label_xx_imag_left_down, label_xx_imag_right_up, label_xx_imag_right_down, label_xx_imag_mid = corp(
            label_xx_imag, 128)
        label_yy_real_left_up, label_yy_real_left_down, label_yy_real_right_up, label_yy_real_right_down, label_yy_real_mid = corp(
            label_yy_real, 128)
        label_yy_imag_left_up, label_yy_imag_left_down, label_yy_imag_right_up, label_yy_imag_right_down, label_yy_imag_mid = corp(
            label_yy_imag, 128)

        np.savez("D:\dataset\B\RA_NF_left_up_%i_%i" % (f, l), xx_real=label_xx_real_left_up,
                 xx_imag=label_xx_imag_left_up, yy_real=label_yy_real_left_up, yy_imag=label_yy_imag_left_up)

        np.savez("D:\dataset\B\RA_NF_left_down_%i_%i" % (f, l), xx_real=label_xx_real_left_down,
                 xx_imag=label_xx_imag_left_down, yy_real=label_yy_real_left_down, yy_imag=label_yy_imag_left_down)

        np.savez("D:\dataset\B\RA_NF_right_up_%i_%i" % (f, l), xx_real=label_xx_real_right_up,
                 xx_imag=label_xx_imag_right_up, yy_real=label_yy_real_right_up, yy_imag=label_yy_imag_right_up)

        np.savez("D:\dataset\B\RA_NF_right_down_%i_%i" % (f, l), xx_real=label_xx_real_right_down,
                 xx_imag=label_xx_imag_right_down, yy_real=label_yy_real_right_down, yy_imag=label_yy_imag_right_down)

        np.savez("D:\dataset\B\RA_NF_mid_%i_%i" % (f, l), xx_real=label_xx_real_mid,
                 xx_imag=label_xx_imag_mid, yy_real=label_yy_real_mid, yy_imag=label_yy_imag_mid)

