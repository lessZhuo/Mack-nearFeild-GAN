import copy
import glob
import lzma
import random
import os
import scipy.io as scio
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch as t


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train", xy=0):
        # self.center_crop = crop_size
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
        self.xy = xy

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        # image_A = Image.open(self.files_A[index % len(self.files_A)])
        # matrix_A = pd.read_csv(lzma.open(self.files_A[index % len(self.files_A)], mode='rb'), header=None)
        # result_A = np.array(matrix_A)  # 生成np 的array
        # image_A = Image.fromarray(np.uint8(result_A*255))  # 从数据，生成image对象
        mask = self.files_A[index]
        label = self.files_B[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        # img = Image.open(img)
        mask = self.read_mask(mask)
        # print(type(mask[0][0]))

        f = mask.shape[0]
        l = mask.shape[1]
        mask = np.rot90(mask, 1)
        label = self.read_csv(label, f, l)

        # mask = mask[np.newaxis, :]
        # print(mask.shape)
        # mask.mean(0)
        # label.mean(0)
        mask = self.center_crop(mask, f, l)
        label = self.center_crop(label, f, l)
        # print(mask.shape)
        # print(mask.mean(0).shape)

        # label = label.astype(np.float32)
        # mask = t.from_numpy(mask)
        # label = t.from_numpy(label)
        # mask = mask.float()

        mask = np.array(mask)
        label = np.array(label)

        result_A = mask[:, :]#, np.newaxis]
        result_B = label[:, :]#, np.newaxis]
        result_A = result_A.astype(np.float)
        result_B = result_B.astype(np.float)

        item_A = self.transform(result_A)
        item_B = self.transform(result_B)
        item_A = item_A.type(torch.FloatTensor)
        item_B = item_B.type(torch.FloatTensor)
        # image_A = np.array(image_A)
        # image_B = np.array(image_B)
        # item_A = t.from_numpy(image_A)
        # item_B = t.from_numpy(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

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
        # print(numpy_mask.shape)

        return numpy_mask

    def read_csv(self, label, f, l, xy=1):
        matrix = pd.read_csv(
            label,
            header=5, usecols=[0], names=['index'])
        # usecols=[0],names=['index'],
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
            # print(mat)

            k = np.stack(mat)
            ma.append(k)
            mk = np.stack(ma, axis=0)
            # print(k.shape, mk.shape)
        xx_real = mk[0] * np.cos((mk[4] / 180) * np.pi)
        xx_imag = mk[0] * np.sin((mk[4] / 180) * np.pi)
        yy_real = mk[3] * np.cos((mk[7] / 180) * np.pi)
        yy_imag = mk[3] * np.sin((mk[7] / 180) * np.pi)
        xx_final = np.array([xx_imag, xx_real])
        yy_final = np.array([yy_imag, yy_real])
        # print(xx_final)

        if xy == 0:
            label = xx_real
        elif xy == 1:
            label = xx_imag
        elif xy == 2:
            label = yy_real
        elif xy == 3:
            label = yy_imag
        return label  # yy_final

    def center_crop(self, numpy_data, f, l):
        """裁剪输入的图片和标签大小"""
        v = 300
        h = 300
        v = (v // l + 1) if (v // l) % 2 == 0 else (v // l + 2)
        h = (h // f + 1) if (h // f) % 2 == 0 else (h // f + 2)
        for i in range(v):
            if i == 0:
                num = numpy_data
            else:
                num = np.append(num, numpy_data, axis=0)
            if i == (v - 1):
                for j in range(h):
                    if j == 0:
                        num_v = num
                    else:
                        num_v = np.append(num_v, num, axis=1)
        num_r = num_v.shape[0]
        num_v1 = num_v.shape[1]

        r1 = int(num_r / 2 - 256 / 2)
        r2 = int(num_r / 2 + 256 / 2)
        v1 = int(num_v1 / 2 - 256 / 2)
        v2 = int(num_v1 / 2 + 256 / 2)
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
