import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import LoadDataset
import FCN
from train import ModelTrainer
from common_tools import plot_line
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time

since = time.time()
if __name__ == "__main__":

    BASE_DIR = r'../datasets/crop_256/new'

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join("../results/fcn", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    crop_size = (300, 300)
    Epoch = 50
    BATCH_SIZE = 50
    LR = 0.1
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    milestones = [150, 225]  #

    transforms_ = [
        transforms.Normalize(mean=[0.0062, 0.0048], std=[1.0016, 1.0003])
    ]

    # ============================ step 1/5 数据 ============================

    # # 构建MyDataset实例
    Load_train = LoadDataset(BASE_DIR, transforms_=transforms_, mode="train", combine=True, direction="x", part="real")
    Load_val = LoadDataset(BASE_DIR, transforms_=transforms_, mode="test", combine=True, direction="x", part="real")

    # 构建DataLoder
    train_data = DataLoader(Load_train, BATCH_SIZE, num_workers=2)
    val_data = DataLoader(Load_val, 12, num_workers=2)

    # bw = False
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

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.MSELoss().to(device)
    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    optimizer = optim.Adam(fcn.parameters(), lr=5e-4)

    # optimizer = optim.SGD(resnet_model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)  # 选择优化器
    #
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)

    # ============================ step 5/5 训练 ============================
    loss_rec = {"train": [], "valid": []}

    best_acc, best_epoch = 0, 0
    best_loss = 0.001
    for epoch in range(start_epoch + 1, Epoch):

        print('Epoch is [{}/{}]'.format(epoch + 1, Epoch))
        if epoch % 30 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train = ModelTrainer.train(fcn, train_data, criterion, optimizer, epoch, device, bw)
        loss_valid = ModelTrainer.evaluate(fcn, val_data, criterion, epoch, device, log_dir, bw)
        print("Epoch[{:0>3}/{:0>3}]  Train loss:{:.8f} Valid loss:{:.8f} LR:{}".format(
            epoch + 1, Epoch, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        # 绘图
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)

        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        if bw:
            sst = 'bw'
        else:
            sst = 'fw'
        if epoch > (Epoch / 2) and loss_valid < best_loss:
            best_loss = loss_valid
            best_epoch = epoch
            path_checkpoint = os.path.join(log_dir, '{}_{}.pth'.format(epoch, sst))
            torch.save(fcn.state_dict(), path_checkpoint)

    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S'),
                                                               best_loss, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    print(time_str)
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
