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

from models_0 import *
from datasets_0 import *
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
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()
Epoch = opt.n_epochs
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_Vail = torch.nn.MSELoss().to(device)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()


G_AB.load_state_dict(torch.load(r"E:\hello pytorch\My_Code\results\test\G_AB_xx_real_0.0000046.pth"))


# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

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

# Training data loader
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),  # unaligned=True
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="test"),
    batch_size=2,
    shuffle=True,
    num_workers=1,
)


# now_time = datetime.datetime.now()
# time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
# log_dir = os.path.join("F:/JZJ/hello pytorch/My_Code", "results", time_str)


def plot_line(train_x, train_y, valid_z, valid_k, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.figure(figsize=(10, 10), dpi=300)
    plt.subplot(211)
    plt.plot(train_x, train_y, label='time', color='b', marker='o', markerfacecolor='b', markersize=10)
    # # plt.plot(valid_z, valid_k, label='Valid', fontsize=100)
    #
    # plt.ylabel(str(mode), fontsize=100)
    # # plt.xlabel('Epoch', fontsize=100)
    plt.tick_params(labelsize=20)
    # # location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc='best', prop={'size': 30})
    # # plt.gca().axes.get_xaxis().set_visible(False)
    plt.title('xx_real loss', fontsize=30)

    plt.subplot(212)

    # plt.ylabel(str('Valid Loss'))
    plt.xlabel('Epoch', fontsize=20)

    # plt.title('Valid Loss', fontsize=10)
    plt.plot(valid_z, valid_k, label='test loss', color='r', marker='o', markerfacecolor='r', markersize=10)
    plt.tick_params(labelsize=20)
    # location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc='best', prop={'size': 30})
    # plt.gca().axes.get_xaxis().set_visible(True)
    # plt.figure(figsize=(14, 14), dpi=300)
    plt.savefig(os.path.join(out_dir, mode + '.tiff'))
    plt.close()


# ----------
#  Training
# ----------

if __name__ == '__main__':
    loss_rec = {"time": [], "test_loss": []}
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(r"E:\hello pytorch\My_Code", r"results\test", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for epoch in range(opt.epoch, opt.n_epochs):
        # --------------
        #  vail Progress
        # --------------

        net = G_AB.eval()
        eval_loss = []
        times = []
        best_times = 1
        # 验证用论文的函数验证 MSE验证
        for j, sample in enumerate(val_dataloader):
            real_A = Variable(sample['A'].to(device))

            real_B = Variable(sample['B'].to(device))

            prev_time = time.time()
            fake_B = net(real_A)
            curr_time = time.time()

            loss = criterion_Vail(fake_B, real_B)
            eval_loss.append(loss.item())
            # If at sample interval save image
            times.append(curr_time - prev_time)

            # sample_images(batches_done)
            real_A = make_grid(real_A, nrow=2, normalize=True)
            real_B = make_grid(real_B, nrow=2, normalize=True)
            # fake_A = make_grid(fake_A, nrow=5, normalize=True)
            fake_B = make_grid(fake_B, nrow=2, normalize=True)
            # Arange images along y-axis
            # image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
            image_grid = torch.cat((real_A, real_B, fake_B), 1)
            # save_image(image_grid, "%s/%s.png" % ("F:/JZJ/hello pytorch/My_Code/results/time_str", batches_done), normalize=False)
            image_grid = image_grid.cpu()  # .transpose(0, 3, 1, 2)
            image_grid = image_grid[0, :, :].detach().numpy()  # [0, 0, :, :]
            # image_grid = Image.fromarray(np.uint8(image_grid * 255)) #.convert('RGB')
            # image_grid = np.array(image_grid)

            plt.figure(figsize=(14, 14), dpi=300)

            # y 轴不可见
            # plt.gca().axes.get_yaxis().set_visible(False)
            # fig, ax = plt.subplots()
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 输出图像#边框设置
            plt.imshow(image_grid)
            # plt.axis('off')
            plt.gca().xaxis.set_ticks_position('top')
            plt.tick_params(labelsize=30)
            # plt.show()
            # ax = plt.subplot(111)
            # ax.invert_yaxis()  # y轴反向
            # ax.set_title('xx_real', fontsize=20)
            plt.title('xx_real', fontsize=30, y=-0.1)
            # plt.colorbar()  # fraction=0.05, pad=0.05
            cb = plt.colorbar()  # fraction=0.05, pad=0.05
            cb.ax.tick_params(labelsize=30)

            batches_done = epoch * len(val_dataloader) + j
            if batches_done % opt.sample_interval == 0:
                number = batches_done
                # plt.savefig("%s/%s.tiff" % (r"E:\hello pytorch\My_Code\results\test\time_str", number),
                #             bbox_inches='tight')
                plt.savefig('%s/%i.tiff' % (log_dir, number), bbox_inches='tight')
            plt.close()

        eval_mean = np.mean(eval_loss)
        time_mean = np.mean(times)
        if time_mean < best_times:
            best_times = time_mean
        print("Epoch[{:0>3}/{:0>3}]  time:{:.6f}  loss_valid:{:.9f} ".format(
            epoch, Epoch, time_mean, eval_mean))

        # 绘图
        loss_rec["time"].append(time_mean), loss_rec["test_loss"].append(eval_mean),

        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["time"], plt_x, loss_rec["test_loss"],
                  mode="loss", out_dir=log_dir)

    print(
        " done ~~~~ {}, best time: {}  ".format(datetime.datetime.strftime(datetime.datetime.now(), '%m-%d_%H-%M'),
                                                best_times))
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
