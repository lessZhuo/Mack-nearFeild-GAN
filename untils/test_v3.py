import torch as t
import numpy as np
from torchvision import transforms
from dataset_v2 import LoadDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

transforms_ = [
    # transforms.Normalize(mean=[0.0062, 0.0048], std=[1.0016, 1.0003])
    transforms.Normalize(mean=[0.193, 0.195,0.195,0.195,], std=[0.927, 1.378])
]

if __name__ == '__main__':
    device = t.device('cpu')
    train = LoadDataset(r"../datasets/crop_128/final",
                        transforms_=transforms_, mode="train")
    train_data = DataLoader(train, batch_size=1, shuffle=True, num_workers=0)
    nb_samples = len(train_data)
    channel_mean = t.zeros(8)
    channel_std = t.zeros(8)
    for i, sample in enumerate(train_data):
        # 载入数据

        data = Variable(sample['B'].to(device))


        N, C, H, W = data.shape[:4]
        data = data.view(N, C, -1)
        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)



    #获取同一batch的均值和标准差
    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)

    # 这是归一化的 mean 和std

    # 这是反归一化的 mean 和std
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]

    print(MEAN)
    print(STD)
