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
        transforms.Normalize(mean=[0.193, 0.195], std=[0.927, 1.378])
    ]

    de_transforms_ = [
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize(mean=[-0.2082, -0.1415], std=[1.0787, 0.7257]),
    ]

    # ============================ step 1/5 数据 ============================

    # # 构建MyDataset实例
    Load_train = LoadDataset(BASE_DIR, transforms_=transforms_, mode="test", combine=True, direction="x", part="real")

    # 构建DataLoder
    data = DataLoader(Load_train, BATCH_SIZE, num_workers=2)

    # ============================ step 2/5 模型 ============================
    fcn = FCN.FCN(1, 2)

    fcn.load_state_dict(torch.load(r"C:\Users\Administrator\PycharmProjects\Mack-nearFeild-GAN\results\fcn\49.pth", map_location=torch.device('cpu')))
    fcn = fcn.to(device)

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.MSELoss().to(device)

    tf = transforms.Compose(de_transforms_)

    rec = {"loss": [], 'time': []}

    for epoch in range(start_epoch + 1, Epoch):
        net = fcn.eval()
        eval_loss = []
        times = []

        prec_time = datetime.now()
        for j, sample in enumerate(data):
            valImg = Variable(sample['A'].to(device))
            valLabel = Variable(sample['B'].to(device))

            prev_time = time.time()
            out = net(valImg)
            curr_time = time.time()

            times.append(curr_time - prev_time)

            loss = criterion(out, valLabel)
            eval_loss.append(loss.item())
            if j % 100 == 0:
                out_mask = valImg[0, 0, :, :].cpu().detach().numpy()
                label_r = tf(valLabel)[0, 0, :, :].cpu().detach().numpy()
                label_i = tf(valLabel)[0, 1, :, :].cpu().detach().numpy()
                out_label_r = tf(out)[0, 0, :, :].cpu().detach().numpy()
                out_label_i = tf(out)[0, 1, :, :].cpu().detach().numpy()

                plt.figure(figsize=(14, 14), dpi=300)

                x1 = plt.subplot(2, 3, 1)
                plt.imshow(out_mask)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x1.set_title('mask')

                x2 = plt.subplot(2, 3, 2)
                plt.imshow(out_label_r)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x2.set_title('out_NF_r')

                x3 = plt.subplot(2, 3, 3)
                plt.imshow(out_label_i)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x3.set_title('out_NF_i')

                x3 = plt.subplot(2, 3, 6)
                plt.imshow(label_i)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x3.set_title('NF_i')

                x3 = plt.subplot(2, 3, 5)
                plt.imshow(label_r)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x3.set_title('NF_r')

                plt.subplots_adjust(wspace=0.4, hspace=0.05)
                plt.savefig('%s/%i_%i.png' % (log_dir, epoch, j), bbox_inches='tight')
                plt.close()

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
