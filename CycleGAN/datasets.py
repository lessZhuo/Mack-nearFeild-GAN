import glob
import lzma
import random
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        # 获取该文件夹所有符合模式匹配格式的文件，变成list返回
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*.*"))

    def __getitem__(self, index):
        matrix_A = pd.read_csv(self.files_A[index % len(self.files_A)], header=None)
        result_A = np.array(matrix_A)  # 生成np 的array
        # image_A = Image.fromarray(np.uint8(result_A*255))  # 从数据，生成image对象

        # image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            # image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            matrix_B = pd.read_csv(self.files_B[random.randint(0, len(self.files_B) - 1)],
                                   header=None)
            result_B = np.array(matrix_B)  # 生成np 的array
            # image_B = Image.fromarray(np.uint8(result_B*255))  # 从数据，生成image对象
        else:
            # image_B = Image.open(self.files_B[index % len(self.files_B)])
            matrix_B = pd.read_csv(self.files_B[index % len(self.files_B)], header=None)
            result_B = np.array(matrix_B)  # 生成np 的array
            # image_B = Image.fromarray(np.uint8(result_B*255))  # 从数据，生成image对象

        # Convert grayscale images to rgb
        # if image_A.mode != "RGB":
        #     image_A = to_rgb(image_A)
        # if image_B.mode != "RGB":
        #     image_B = to_rgb(image_B)
        result_A = result_A[:,:,np.newaxis]
        result_B = result_B[:,:,np.newaxis]
        # print(result_A.shape)
        # print(result_B.shape)
        result_A = result_A.astype(np.float)
        result_B = result_B.astype(np.float)
        item_A = self.transform(result_A)
        item_B = self.transform(result_B)
        item_A=item_A.type(torch.FloatTensor)
        item_B = item_B.type(torch.FloatTensor)
        # print(item_A.shape)
        # print(item_B.shape)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
