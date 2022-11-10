import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo_test", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")  # 8
parser.add_argument("--img_height", type=int, default=256, help="size of image height")  # 256
parser.add_argument("--img_width", type=int, default=256, help="size of image width")  # 256
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--input_channels", type=int, default=1, help="number of input channels")
parser.add_argument("--output_channels", type=int, default=2, help="number of image channels")
parser.add_argument("--proportion", type=float, default=0.9, help="proportion of A to B loss in total loss ")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("../result/cycleGan/images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("../result/cycleGan/saved_models/%s" % opt.dataset_name, exist_ok=True)


cuda = torch.cuda.is_available()
Epoch = opt.n_epochs
input_shape = (opt.input_channels, opt.img_height, opt.img_width)
output_shape = (opt.output_channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, output_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(output_shape, input_shape, opt.n_residual_blocks)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_Vail = torch.nn.MSELoss().to(device)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()

# ---------------------------------设置加载的参数---------------------------------------------
G_AB.load_state_dict(torch.load(r"E:\hello pytorch\My_Code\results\test\G_AB_xx_real_0.0000046.pth"))
G_BA.load_state_dict(torch.load(r"E:\hello pytorch\My_Code\results\test\G_AB_xx_real_0.0000046.pth"))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

image_save_plot = ImagePlotSave(output_shape, input_shape)

# Image transformations
transforms_ = [
    # transforms.Resize(int(opt.img_height), Image.BICUBIC),  # opt.img_height * 1.12
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.Normalize([0.5], [0.5]),
    transforms.Normalize([1], [1]),
]

#  data loader
dataloader = DataLoader(
    MaskNfDataset("../datasets/crop_256", transforms_=transforms_, combine=True, direction="x"),  # unaligned=True
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

if __name__ == '__main__':
    loss_rec = {"time_AB": [], "time_BA": [], "G_AB_loss": [], "G_BA_loss": []}
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(r"..\results\cycleGan\test", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for epoch in range(opt.epoch, opt.n_epochs):
        # --------------
        #  vail Progress
        # --------------

        net_G_AB = G_AB.eval()
        net_G_BA = G_BA.eval()
        eval_loss_AB = []
        eval_loss_BA = []
        times_AB = []
        times_BA = []
        best_times = 1
        # 验证用论文的函数验证 MSE验证
        for j, sample in enumerate(dataloader):
            # -------------读取数据---------------------
            real_A = Variable(sample['A'].to(device))
            real_B = Variable(sample['B'].to(device))

            # ------------计算A to B的-----------------
            prev_time = time.time()
            fake_B = net_G_AB(real_A)
            curr_time = time.time()
            times_AB.append(curr_time - prev_time)
            loss = criterion_Vail(fake_B, real_B)
            eval_loss_AB.append(loss.item())

            # ------------计算B to A的-----------------
            prev_time = time.time()
            fake_A = net_G_AB(real_B)
            curr_time = time.time()
            times_BA.append(curr_time - prev_time)
            loss = criterion_Vail(fake_A, real_A)
            eval_loss_BA.append(loss.item())

            batches_done = epoch * len(dataloader) + j
            if batches_done % opt.sample_interval == 0:
                image_save_plot.sample_images(epoch, batches_done, log_dir, real_A=real_A, real_B=real_B, fake_A=fake_A,
                                              fake_B=fake_B)

        eval_mean_AB = np.mean(eval_loss_AB)
        time_mean_AB = np.mean(times_AB)

        eval_mean_BA = np.mean(eval_loss_BA)
        time_mean_BA = np.mean(times_BA)
        if time_mean_AB < best_times:
            best_times = time_mean_AB
        print("Epoch[{:0>3}/{:0>3}]  time_AB:{:.6f}  loss_AB_valid:{:.9f}  time_BA:{:.6f}  loss_BA_valid:{:.9f} ".format(
            epoch, Epoch, time_mean_AB, eval_mean_AB, time_mean_BA, eval_mean_BA))

        # 绘图
        loss_rec["time_AB"].append(time_mean_AB), loss_rec["test_loss_AB"].append(eval_mean_AB),
        loss_rec["time_BA"].append(time_mean_BA), loss_rec["test_loss_BA"].append(eval_mean_BA)

        plt_x = np.arange(1, epoch + 2)
        image_save_plot.plot_line(plt_x, loss_rec["time_AB"], loss_rec["test_loss_AB"], loss_rec["time_BA"],
                                  loss_rec["test_loss_BA"],
                                  out_dir=log_dir)



    print(
        " done ~~~~ {}, best time: {}  ".format(datetime.datetime.strftime(datetime.datetime.now(), '%m-%d_%H-%M'),
                                                best_times))
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
