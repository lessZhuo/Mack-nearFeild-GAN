import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import time
from dataset import LoadDataset
import FCN
from train import ModelTrainer
from common_tools import plot_line_v2
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from train import save_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    BASE_DIR = r'../datasets/crop_256/new'

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join("../results/fcn/test", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    crop_size = (300, 300)
    Epoch = 50
    BATCH_SIZE = 1
    LR = 0.1
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    milestones = [150, 225]  #

    transforms_ = [
        # transforms.Normalize(mean=[0.193, 0.195], std=[0.927, 1.378])
        transforms.Normalize(mean=[0.0062, 0.0048], std=[1.0016, 1.0003])
    ]

    de_transforms_ = [
        # transforms.Normalize(mean=[-0.2082, -0.1415], std=[1.0787, 0.7257]),
        transforms.Normalize(mean=[-0.0062, -0.0048], std=[0.9984, 0.9997])
    ]

    # ============================ step 1/5 数据 ============================

    # # 构建MyDataset实例
    Load_train = LoadDataset(BASE_DIR, transforms_=transforms_, mode="test", combine=True, direction="x", part="real")

    # 构建DataLoder
    data = DataLoader(Load_train, BATCH_SIZE, num_workers=2)

    bw = True
    if bw:
        input_channel = 2
        output_channel = 2
    else:
        input_channel = 1
        output_channel = 2
    # ============================ step 2/5 模型 ============================
    fcn = FCN.FCN(input_channel=input_channel, output_channel=output_channel)
    fcn = fcn.to(device)

    fcn.load_state_dict(
        torch.load(r"C:\Users\Administrator\Desktop\paper_train\fcn\46_bw.pth", map_location=torch.device('cpu')))
    fcn = fcn.to(device)

    # ============================ step 3/5 损失函数 ============================
    if bw:
        criterion = nn.NLLLoss().to(device)
    else:
        criterion = nn.MSELoss().to(device)

    tf = transforms.Compose(de_transforms_)

    rec = {"loss": [], 'time': []}

    for epoch in range(start_epoch + 1, Epoch):
        net = fcn.eval()
        eval_loss = []
        times = []

        prec_time = datetime.now()
        for j, sample in enumerate(data):
            if bw:
                valImg = Variable(sample['B'].to(device))
                valLabel = Variable(sample['C'].to(device))
            else:
                valImg = Variable(sample['A'].to(device))
                valLabel = Variable(sample['B'].to(device))

            prev_time = time.time()
            out = net(valImg)
            if bw:
                out = F.log_softmax(out, dim=1)
            curr_time = time.time()

            times.append(curr_time - prev_time)

            loss = criterion(out, valLabel)
            eval_loss.append(loss.item())

            if bw:
                valImg = tf(valImg)
            else:
                out = tf(out)
                valLabel = tf(valLabel)

            if j % 100 == 0:
                if bw:
                    save_img(valLabel, valImg, out, epoch, j, log_dir, bw)
                else:
                    save_img(valImg, valLabel, out, epoch, j, log_dir, bw)

        eval_mean = np.mean(eval_loss)
        time_mean = np.mean(times)
        rec["loss"].append(eval_mean), rec["time"].append(time_mean)
        print(
            "Epoch[{:0>3}/{:0>3}]  time:{:.6f}  loss:{:.9f} ".format(
                epoch, Epoch, time_mean, eval_mean))

        plt_x = np.arange(1, epoch + 2)
        plot_line_v2(plt_x, rec["loss"], rec["time"], mode="test", out_dir=log_dir)

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    print(time_str)

# print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
