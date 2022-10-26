# -*- coding: utf-8 -*-
"""
# @file name  : 1_split_dataset.py
# @author     : tingsongyu
# @date       : 2019-09-07 10:08:00
# @brief      : 将数据集划分为训练集，验证集，测试集
"""

import os
import random
import shutil


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)

    dataset_dir = os.path.join(r"F:\JZJ\hello pytorch\My_Code\PyTorch-GAN\data\mask_nf_xxyy\mask_nf_xxr")
    split_dir = os.path.join(r"F:\JZJ\hello pytorch\My_Code\PyTorch-GAN\data\monet2photo")
    train_dir = os.path.join(split_dir, "train")
    # valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    train_pct = 0.8
    # valid_pct = 0.1
    test_pct = 0.2

    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:
            print(sub_dir)

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.xz'), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            test_point = int(img_count * test_pct)

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                # elif i < valid_point:
                #     out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, test:{}'.format(sub_dir, train_point, test_point))
