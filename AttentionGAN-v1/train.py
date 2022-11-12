#!/usr/bin/python3

import argparse
import itertools
import sys
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.utils import save_image
from models import *
from utils import *
from datasets import *
import functools

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from utils import print_network

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--save_name', type=str, default='ar_neutral2happiness')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--lambda_cycle', type=int, default=10)
parser.add_argument('--lambda_identity', type=int, default=0)
parser.add_argument('--lambda_a', type=int, default=0)
parser.add_argument('--lambda_b', type=int, default=0)
parser.add_argument('--lambda_pixel', type=int, default=1)
parser.add_argument('--lambda_reg', type=float, default=1e-6)
parser.add_argument('--gan_curriculum', type=int, default=10,
                    help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.01,
                    help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5,
                    help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=3,
                    help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--norm', type=str, default='instance',
                    help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')
parser.add_argument("--img_height", type=int, default=256, help="size of image height")  # 128
parser.add_argument("--img_width", type=int, default=256, help="size of image width")  # 128
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###### Definition of variables ######
# Networks
netG_A2B = ResnetGenerator_Attention(opt.input_nc, opt.output_nc, opt.ngf, n_blocks=9)
netG_B2A = ResnetGenerator_Attention(opt.input_nc, opt.output_nc, opt.ngf, n_blocks=9)
netD_A = NLayerDiscriminator(opt.input_nc, opt.ndf, n_layers=3, norm_type=opt.norm)
netD_B = NLayerDiscriminator(opt.input_nc, opt.ndf, n_layers=3, norm_type=opt.norm)

print('---------- Networks initialized -------------')
print_network(netG_A2B)
print_network(netG_B2A)
print_network(netD_A)
print_network(netD_B)
print('-----------------------------------------------')

input_shape = (opt.input_channels, opt.img_height, opt.img_width)
output_shape = (opt.output_channels, opt.img_height, opt.img_width)

# save image and plot loss line
image_save_plot = ImagePlotSave(output_shape, input_shape)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_Vail = torch.nn.MSELoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(),
                                 lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(),
                                 lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

# fake_A_buffer = ReplayBuffer()
# fake_B_buffer = ReplayBuffer()

fake_A_buffer = ImagePool(opt.pool_size)
fake_B_buffer = ImagePool(opt.pool_size)

# Dataset loader
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Training data loader
dataloader = DataLoader(
    MaskNfDataset("../datasets/crop_256", transforms_=transforms_, combine=True, direction="x"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    MaskNfDataset("../datasets/crop_256", transforms_=transforms_, mode="test", combine=True,
                  direction="x"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


if __name__ == '__main__':
    loss_rec = {"loss_D": [], "loss_G": [], "loss_G_AB_valid": [], "loss_G_AB_train": [], "loss_G_BA_valid": [],
                "loss_G_BA_train": []}
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join("../results/", "cycleGan", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    prev_time = time.time()
    # i, batch = next(enumerate(dataloader))
    best_acc, best_epoch = 0, 0
    best_loss = 0.01

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        D_loss = []
        G_loss = []
        train_loss_AB = []
        train_loss_BA = []
        for i, batch in enumerate(dataloader):
            # 清除缓存
            torch.cuda.empty_cache()
            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # ------------------
            #  Train Generators
            # ------------------

            netG_A2B.train()
            netG_B2A.train()

            fake_B, o1_b, o2_b, o3_b, o4_b, o5_b, o6_b, o7_b, o8_b, o9_b, o10_b, \
            a1_b, a2_b, a3_b, a4_b, a5_b, a6_b, a7_b, a8_b, a9_b, a10_b, \
            i1_b, i2_b, i3_b, i4_b, i5_b, i6_b, i7_b, i8_b, i9_b = netG_A2B(real_A)  # G_A(A)
            rec_A, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _ = netG_B2A(fake_B)  # G_B(G_A(A))

            fake_A, o1_a, o2_a, o3_a, o4_a, o5_a, o6_a, o7_a, o8_a, o9_a, o10_a, \
            a1_a, a2_a, a3_a, a4_a, a5_a, a6_a, a7_a, a8_a, a9_a, a10_a, \
            i1_a, i2_a, i3_a, i4_a, i5_a, i6_a, i7_a, i8_a, i9_a = netG_B2A(real_B)  # G_B(B)
            rec_B, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _ = netG_A2B(fake_A)

            set_requires_grad([netD_A, netD_B], False)

            optimizer_G.zero_grad()

            loss_G_A = criterion_GAN(netD_A(fake_B), target_real)
            # GAN loss D_B(G_B(B))
            loss_G_B = criterion_GAN(netD_B(fake_A), target_real)
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_A = criterion_cycle(rec_A, real_A) * opt.lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_B = criterion_cycle(rec_B, real_B) * opt.lambda_B

            loss_GAN = (loss_G_A + loss_G_B) / 2
            loss_cycle = (loss_cycle_B + loss_cycle_A) / 2
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle
            G_loss.append(loss_G.item())

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            set_requires_grad([netD_A, netD_B], True)

            optimizer_D_A.zero_grad()

            # Real loss
            pred_real_A = netD_A.forward(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, target_real)

            # Fake loss
            fake_A = fake_A_buffer.query(fake_A)
            pred_fake_A = netD_A.forward(fake_A.detach())
            loss_D_fake_A = criterion_GAN(pred_fake_A, target_fake)

            loss_D_A = (loss_D_real_A + loss_D_fake_A) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            # Real loss
            pred_real_B = netD_B.forward(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, target_real)

            # Fake loss
            fake_B = fake_B_buffer.query(fake_B)
            pred_fake_B = netD_B.forward(fake_B.detach())
            loss_D_fake_B = criterion_GAN(pred_fake_B, target_fake)

            loss_D_B = (loss_D_real_B + loss_D_fake_B) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            # Total loss
            loss_D = loss_D_B * opt.proportion + loss_D_A * (1 - opt.proportion)
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
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    # loss_identity.item(),
                    time_left,
                )
            )

            # --------------
            #  vail train Progress
            # --------------

            loss_AB = criterion_Vail(fake_B, real_B)
            train_loss_AB.append(loss_AB.item())

            loss_BA = criterion_Vail(fake_A, real_A)
            train_loss_BA.append(loss_BA.item())

            train_mean_AB = np.mean(train_loss_AB)
            train_mean_BA = np.mean(train_loss_BA)

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                image_save_plot.sample_images(epoch, batches_done, log_dir, real_A=real_A, real_B=real_B, fake_A=fake_A,
                                              fake_B=fake_B)
        print("Epoch[{:0>3}/{:0>3}]  train_AB_loss:{:.6f}  trainBA_loss:{:.6f} ".format(epoch, opt.n_epochs, train_mean_AB,
                                                                                        train_mean_BA))

        # --------------
        #  vail Progress
        # --------------

        net_G_AB_v = netG_A2B.eval()
        net_G_BA_v = netG_B2A.eval()
        eval_loss_G_AB = []
        eval_loss_G_BA = []
        # 验证用论文的函数验证 MSE验证
        for j, sample in enumerate(val_dataloader):
            real_A = Variable(sample['A'].to(device))
            real_B = Variable(sample['B'].to(device))

            fake_B = net_G_AB_v(real_A)
            fake_A = net_G_BA_v(real_B)

            loss_G_AB = criterion_Vail(fake_B, real_B)
            eval_loss_G_AB.append(loss_G_AB.item())

            loss_G_BA = criterion_Vail(fake_A, real_A)
            eval_loss_G_BA.append(loss_G_BA.item())

        G_mean = np.mean(G_loss)
        D_mean = np.mean(D_loss)
        eval_mean_AB = np.mean(eval_loss_G_AB)
        eval_mean_BA = np.mean(eval_loss_G_BA)

        print(
            "Epoch[{:0>3}/{:0>3}]  loss_G:{:.6f} loss_D:{:.6f} loss_G_AB_valid:{:.9f} loss_G_BA_valid:{:.9f} train_G_AB_loss:{:.9f} train_G_BA_loss:{:.9f}".format(
                epoch, opt.n_epochs, G_mean, D_mean, eval_mean_AB, eval_mean_BA, train_mean_AB, train_mean_BA))

        # 绘图
        loss_rec["loss_G"].append(G_mean), loss_rec["loss_D"].append(D_mean),
        loss_rec["loss_G_AB_valid"].append(eval_mean_AB), loss_rec["loss_G_BA_valid"].append(eval_mean_BA), \
        loss_rec["loss_G_AB_train"].append(train_mean_AB), loss_rec["loss_G_BA_train"].append(train_mean_BA)

        plt_x = np.arange(1, epoch + 2)
        image_save_plot.plot_line(plt_x, loss_rec["loss_G"], loss_rec["loss_D"], loss_rec["loss_G_AB_valid"],
                                  loss_rec["loss_G_AB_train"], loss_rec["loss_G_BA_valid"], loss_rec["loss_G_BA_train"],
                                  out_dir=log_dir)
        # plot_line(plt_x, loss_rec["loss_valid"], mode="xx_real loss", out_dir=log_dir)
        # ------------------------------------------temp-------------------------------------
        if epoch > 1:
            image_save_plot.plot_line(plt_x[1:], loss_rec["loss_G"][1:], loss_rec["loss_D"][1:],
                                      loss_rec["loss_G_AB_valid"][1:],
                                      loss_rec["loss_G_AB_train"][1:], loss_rec["loss_G_BA_valid"][1:],
                                      loss_rec["loss_G_BA_train"][1:],
                                      out_dir=log_dir, mark='temp')

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if eval_mean_AB + eval_mean_BA < best_loss:
            best_loss = eval_mean_AB + eval_mean_BA
            best_epoch = epoch
            print("epoch: %i" % epoch)
            print("n_epoch: %i" % opt.n_epochs)
            print("best_loss: %f " % best_loss)

            torch.save(netG_B2A.state_dict(), '%s/G_AB_%d.pth' % (log_dir, epoch))
            torch.save(netG_A2B.state_dict(), '%s/G_BA_%d.pth' % (log_dir, epoch))

    print(
        " done ~~~~ {}, best acc: {} in :{} epochs. ".format(
            datetime.datetime.strftime(datetime.datetime.now(), '%m-%d_%H-%M'),
            best_loss, best_epoch))
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)



            # save_image(torch.cat([
            #     real_A.data.cpu()[0] * 0.5 + 0.5,
            #     mask_B.data.cpu()[0],
            #     fake_B.data.cpu()[0] * 0.5 + 0.5, temp_B.data.cpu()[0] * 0.5 + 0.5], 2),
            #     '%s/%04d_%04d_progress_B.png' % (save_path, epoch + 1, i + 1))
            #
            # save_image(torch.cat([
            #     real_B.data.cpu()[0] * 0.5 + 0.5,
            #     mask_A.data.cpu()[0],
            #     fake_A.data.cpu()[0] * 0.5 + 0.5, temp_A.data.cpu()[0] * 0.5 + 0.5], 2),
            #     '%s/%04d_%04d_progress_A.png' % (save_path, epoch + 1, i + 1))

