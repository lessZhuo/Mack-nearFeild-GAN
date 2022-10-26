import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mask_fcn.dataset import LoadDataset
from mask_fcn import FCN
from mask_fcn.train import ModelTrainer
from mask_fcn.common_tools import  plot_line

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
since = time.time()
if __name__ == "__main__":

    # config
    # train_dir = os.path.join(BASE_DIR, "..", "..", "Data", "cifar-10",  "cifar10_train")
    # test_dir = os.path.join(BASE_DIR, "..", "..", "Data", "cifar-10", "cifar10_test")
    train_1 = r'E:\Datasets\近场数据\RECT_ARRAY_MASK'#train_maks
    train_2 = r"E:\Datasets\近场数据\RECT_ARRAY_NF_x0_y0"#train_label
    val_1 = r'E:\Datasets\近场数据\RECT_ARRAY_MASK'  # train_maks
    val_2= r"E:\Datasets\近场数据\RECT_ARRAY_NF_x0_y0"  # train_label

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(BASE_DIR, "..", "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    crop_size = (300, 300)
    Epoch = 1
    BATCH_SIZE = 50
    LR = 0.1
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    milestones = [150, 225]  #

    # ============================ step 1/5 数据 ============================

    # # 构建MyDataset实例
    Load_train = LoadDataset([train_1, train_2], crop_size,xy=0,mode='train',split_n=0.8)
    Load_val = LoadDataset([val_1, val_2], crop_size,xy=0,mode='valid',split_n=0.8)

    # 构建DataLoder
    train_data = DataLoader(Load_train, BATCH_SIZE, num_workers=2)
    val_data = DataLoader(Load_val, 12, num_workers=2)


    # ============================ step 2/5 模型 ============================
    fcn = FCN.FCN()
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
    best_loss=0.001
    for epoch in range(start_epoch + 1, Epoch):


        print('Epoch is [{}/{}]'.format(epoch + 1, Epoch))
        if epoch % 30 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train = ModelTrainer.train(fcn,train_data, criterion, optimizer, epoch, device)
        loss_valid = ModelTrainer.evaluate(fcn,val_data, criterion,epoch, device,log_dir)
        print("Epoch[{:0>3}/{:0>3}]  Train loss:{:.8f} Valid loss:{:.8f} LR:{}".format(
            epoch + 1, Epoch, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        # 绘图
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)

        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)

        if epoch > (Epoch/2) and loss_valid < best_loss:
            best_loss=loss_valid
            best_epoch = epoch
            path_checkpoint = os.path.join(log_dir, '{}.pth'.format(epoch))
            torch.save(fcn.state_dict(), path_checkpoint)
            # checkpoint = {"model_state_dict": resnet_model.state_dict(),
            #           "optimizer_state_dict": optimizer.state_dict(),
            #           "epoch": epoch,
            #           "best_acc": best_acc}
            #
            # path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            # torch.save(checkpoint, path_checkpoint)

    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S'),
                                                      best_loss, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    print(time_str)
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))