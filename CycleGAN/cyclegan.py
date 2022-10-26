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
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="cyclegan", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")  # 8
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

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    # transforms.Resize(int(opt.img_height)),  # opt.img_height * 1.12
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    batch_size=1,
    shuffle=True,
    num_workers=1,
)


# now_time = datetime.datetime.now()
# time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
# log_dir = os.path.join("F:/JZJ/hello pytorch/My_Code", "results", time_str)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=1, normalize=True)
    real_B = make_grid(real_B, nrow=1, normalize=True)
    fake_A = make_grid(fake_A, nrow=1, normalize=True)
    fake_B = make_grid(fake_B, nrow=1, normalize=True)
    # Arange images along y-axis
    # image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    # save_image(image_grid, "%s/%s.png" % ("F:/JZJ/hello pytorch/My_Code/results/time_str", batches_done), normalize=False)

    image_grid = image_grid.cpu()  # .transpose(0, 3, 1, 2)
    image_grid = image_grid[0, :, :].detach().numpy()  # [0, 0, :, :]

    # image_grid = Image.fromarray(np.uint8(image_grid * 255)) #.convert('RGB')
    # image_grid = np.array(image_grid)
    # fake_B = fake_B.cpu()  # .transpose(0, 3, 1, 2)
    # fake_B = fake_B[0, :, :].detach().numpy()  # [0, 0, :, :]

    # plt.figure(figsize=(14, 14), dpi=300)

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
    plt.title('xx_imag', fontsize=50, y=-0.1)
    cb = plt.colorbar()  # fraction=0.05, pad=0.05
    cb.ax.tick_params(labelsize=30)

    plt.savefig("%s/%s.jpg" % ("../../results/time_str", batches_done), bbox_inches='tight')
    plt.close()


def plot_line(train_x, train_y, valid_x, valid_y, valid_z, valid_k, train_m, train_n, mode, out_dir):
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
    plt.figure(figsize=(14, 14), dpi=300)
    plt.subplot(211)
    plt.plot(train_x, train_y, label='lossG',
             color='b', marker='o', markerfacecolor='b', markersize=10)
    plt.plot(valid_x, valid_y, label='lossD',
             color='r', marker='o', markerfacecolor='r', markersize=10)
    plt.tick_params(labelsize=30)
    # # plt.plot(valid_z, valid_k, label='Valid')
    #
    # plt.ylabel(str(mode))
    # # plt.xlabel('Epoch')

    # location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc='best', prop={'size': 30})
    # plt.legend(loc='best')
    # plt.gca().axes.get_xaxis().set_visible(False)
    plt.title('xx_imag loss', fontsize=30)

    plt.subplot(212)
    # plt.ylabel(str('Valid Loss'))
    plt.xlabel('Epoch', fontsize=30)
    # plt.title('Valid Loss', fontsize=100)
    plt.plot(valid_z, valid_k, label='Valid',
             color='b', marker='o', markerfacecolor='b', markersize=10)
    # plt.scatter(valid_z, valid_k, color='', marker='o', edgecolors='b', s=200)
    plt.plot(train_m, train_n, label='Train',
             color='r', marker='o', markerfacecolor='r', markersize=10)

    plt.tick_params(labelsize=30)

    # plt.scatter(train_m, train_n, color='', marker='o', edgecolors='r', s=200)
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
    loss_rec = {"loss_D": [], "loss_G": [], "loss_valid": [], "loss_train": []}
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join("../../", "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    prev_time = time.time()
    i, batch = next(enumerate(dataloader))
    best_acc, best_epoch = 0, 0
    best_loss = 0.0005
    for epoch in range(opt.epoch, opt.n_epochs):
        D_loss = []
        G_loss = []
        train_loss = []
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))*0.9), requires_grad=False)
            fake = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))*0.1), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            # ll = G_BA(real_A)
            # print(ll.shape)
            # print(real_A.shape)
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
            G_loss.append(loss_G.item())
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2
            D_loss.append(loss_D.item())

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # --------------
        #  vail train Progress
        # --------------

            net = G_AB.train()
        # train_loss = []

        # 验证用论文的函数验证 MSE验证
        # for j, sample in enumerate(dataloader):
        #     real_A = Variable(sample['A'].to(device))
        #     real_B = Variable(sample['B'].to(device))

            fake_B = net(real_A)

            loss = criterion_Vail(fake_B, real_B)
            train_loss.append(loss.item())

        # G_mean = np.mean(G_loss)
        # D_mean = np.mean(D_loss)
            train_mean = np.mean(train_loss)

        print("Epoch[{:0>3}/{:0>3}]  train_loss:{:.6f}".format(epoch, Epoch, train_mean))

        # --------------
        #  vail Progress
        # --------------

        net = G_AB.eval()
        eval_loss = []
        # 验证用论文的函数验证 MSE验证
        for j, sample in enumerate(val_dataloader):
            real_A = Variable(sample['A'].to(device))
            real_B = Variable(sample['B'].to(device))

            fake_B = net(real_A)

            loss = criterion_Vail(fake_B, real_B)
            eval_loss.append(loss.item())

        G_mean = np.mean(G_loss)
        D_mean = np.mean(D_loss)
        eval_mean = np.mean(eval_loss)

        print("Epoch[{:0>3}/{:0>3}]  loss_G:{:.6f} loss_D:{:.6f} loss_valid:{:.6f} train_loss:{:.6f}".format(
            epoch, Epoch, G_mean, D_mean, eval_mean, train_mean))

        # 绘图
        loss_rec["loss_G"].append(G_mean), loss_rec["loss_D"].append(D_mean),
        loss_rec["loss_valid"].append(eval_mean), loss_rec["loss_train"].append(train_mean)

        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["loss_G"], plt_x, loss_rec["loss_D"],
                  plt_x, loss_rec["loss_valid"], plt_x, loss_rec["loss_train"], mode="loss", out_dir=log_dir)
        # plot_line(plt_x, loss_rec["loss_valid"], mode="xx_real loss", out_dir=log_dir)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # 保存符合要求的模型信息
        if epoch != -1 and eval_mean < best_loss:
            best_loss = eval_mean
            best_epoch = epoch
            torch.save(G_AB.state_dict(), '../../results/save_model/G_AB_xx_imag_%d.pth' % epoch)

    print(" done ~~~~ {}, best acc: {} in :{} epochs. "
          .format(datetime.datetime.strftime(datetime.datetime.now(), '%m-%d_%H-%M'), best_loss, best_epoch))
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
