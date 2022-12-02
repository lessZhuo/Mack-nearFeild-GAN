import random
import time
import datetime
import sys
import os
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class ImagePlotSave:
    def __init__(self, output_shape, input_shape):
        self.output_shape = output_shape
        self.input_shape = input_shape

    # 保存图片
    def sample_images(self, epoch, batches_done, log_dir, real_A, fake_A, real_B, fake_B):
        """Saves a generated sample from the test set"""

        real_B_r = real_B[0, 0, :, :]
        real_B_i = real_B[0, 1, :, :]
        fake_B_r = fake_B[0, 0, :, :]
        fake_B_i = fake_B[0, 1, :, :]
        real_A = real_A[0, 0, :, :]
        fake_A = fake_A[0, 0, :, :]

        # --------------------------2022.11.06 add ---------------------------
        real_B_r = real_B_r.cpu().squeeze().detach().numpy()
        real_B_i = real_B_i.cpu().squeeze().detach().numpy()
        fake_B_r = fake_B_r.cpu().squeeze().detach().numpy()
        fake_B_i = fake_B_i.cpu().squeeze().detach().numpy()
        real_A = real_A.cpu().squeeze().detach().numpy()
        fake_A = fake_A.cpu().squeeze().detach().numpy()
        plt.figure(figsize=(14, 14), dpi=300)
        x1 = plt.subplot(2, 3, 1)
        plt.imshow(real_A)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 0)
        x1.set_title('Mask')
        x2 = plt.subplot(2, 3, 2)
        plt.imshow(fake_B_r)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 1)
        x2.set_title('Generated real part of NF')
        x3 = plt.subplot(2, 3, 3)
        plt.imshow(fake_B_i)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x3.set_title('Generated imaginary part of NF')
        x4 = plt.subplot(2, 3, 4)
        plt.imshow(fake_A)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 0)
        x4.set_title('Generated_Mask')
        x5 = plt.subplot(2, 3, 5)
        plt.imshow(real_B_r)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 1)
        x5.set_title('real part of NF')
        x6 = plt.subplot(2, 3, 6)
        plt.imshow(real_B_i)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x6.set_title('imaginary part of NF')

        plt.subplots_adjust(wspace=0.4, hspace=0.05)
        plt.savefig('%s/%i_%i.png' % (log_dir, epoch, batches_done), bbox_inches='tight')
        plt.close()

    def plot_line(selft, epoch, loss_G, loss_D, valid_G_AB, train_G_AB, valid_G_BA, train_G_BA, out_dir, mark='all'):
        """
        绘制训练和验证集的loss曲线
        """
        plt.figure(figsize=(28, 14), dpi=300)
        plt.subplot(221)
        plt.plot(epoch, loss_G, label='lossG', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, loss_D, label='lossD', color='r', marker='o', markerfacecolor='r', markersize=15)
        # # plt.plot(valid_z, valid_k, label='Valid', fontsize=100)
        #
        plt.xlabel('Epoch', fontsize=30)
        plt.tick_params(labelsize=30)
        # # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        # # plt.gca().axes.get_xaxis().set_visible(False)
        plt.title('loss', fontsize=30)

        plt.subplot(222)

        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, valid_G_AB, label='Valid', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, train_G_AB, label='Train', color='r', marker='o', markerfacecolor='r', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('G_AB', fontsize=30)

        plt.subplot(223)

        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, valid_G_BA, label='Valid', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, train_G_BA, label='Train', color='r', marker='o', markerfacecolor='r', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('G_BA', fontsize=30)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.savefig(os.path.join(out_dir, mark + r'loss.png'))
        plt.close()

    def plot_line_test(selft, epoch, time_mean_AB, eval_mean_AB, eval_mean_AB_r, eval_mean_AB_i, time_mean_BA,
                       eval_mean_BA, out_dir):
        """
        绘制训练和验证集的loss曲线
        """
        plt.figure(figsize=(28, 14), dpi=300)
        plt.subplot(221)
        plt.plot(epoch, time_mean_AB, label='time_mean_AB', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, time_mean_BA, label='time_mean_BA', color='r', marker='o', markerfacecolor='r', markersize=15)

        #
        plt.xlabel('Epoch', fontsize=30)
        plt.tick_params(labelsize=30)
        plt.legend(loc='best', prop={'size': 30})
        plt.title('time', fontsize=30)

        plt.subplot(222)

        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, eval_mean_AB, label='G_AB_loss', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, eval_mean_AB_r, label='G_BA_loss_r', color='r', marker='o', markerfacecolor='r', markersize=15)
        plt.plot(epoch, eval_mean_AB_i, label='G_BA_loss_i', color='y', marker='o', markerfacecolor='y', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('loss_AB', fontsize=30)

        plt.subplot(223)

        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, eval_mean_BA, label='G_BA_loss', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('loss_BA', fontsize=30)

        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        plt.savefig(os.path.join(out_dir, r'test_result.png'))
        plt.close()

    def sample_images_v2(self, epoch, batches_done, log_dir, real_A, fake_A, real_B, fake_B):
        """Saves a generated sample from the test set"""

        real_B_r = real_B[0, 0, :, :]
        real_B_i = real_B[0, 1, :, :]
        fake_B_r = fake_B[0, 0, :, :]
        fake_B_i = fake_B[0, 1, :, :]
        real_A = real_A[0, :, :, :].max(dim=0)[1].data
        fake_A = fake_A[0, :, :, :].max(dim=0)[1].data

        real_B_r = real_B_r.cpu().squeeze().detach().numpy()
        real_B_i = real_B_i.cpu().squeeze().detach().numpy()
        fake_B_r = fake_B_r.cpu().squeeze().detach().numpy()
        fake_B_i = fake_B_i.cpu().squeeze().detach().numpy()
        real_A = real_A.cpu().squeeze().detach().numpy()
        fake_A = fake_A.cpu().squeeze().detach().numpy()
        plt.figure(figsize=(14, 14), dpi=300)
        x1 = plt.subplot(2, 3, 1)
        plt.imshow(real_A)
        plt.colorbar(fraction=0.05, pad=0.05)
        plt.clim(0, 1)
        x1.set_title('Mask')
        x2 = plt.subplot(2, 3, 2)
        plt.imshow(fake_B_r)
        plt.colorbar(fraction=0.05, pad=0.05)
        plt.clim(-1, 1)
        x2.set_title('Generated real part of NF')
        x3 = plt.subplot(2, 3, 3)
        plt.imshow(fake_B_i)
        plt.colorbar(fraction=0.05, pad=0.05)
        plt.clim(-1.001, -0.95)
        x3.set_title('Generated imaginary part of NF')
        x4 = plt.subplot(2, 3, 4)
        plt.imshow(fake_A)
        plt.colorbar(fraction=0.05, pad=0.05)
        plt.clim(0, 1)
        x4.set_title('Generated_Mask')
        x5 = plt.subplot(2, 3, 5)
        plt.imshow(real_B_r)
        plt.colorbar(fraction=0.05, pad=0.05)
        plt.clim(-1, 1)
        x5.set_title('real part of NF')
        x6 = plt.subplot(2, 3, 6)
        plt.imshow(real_B_i)
        plt.colorbar(fraction=0.05, pad=0.05)
        plt.clim(-1.001, -0.95)
        x6.set_title('imaginary part of NF')

        plt.subplots_adjust(wspace=0.4, hspace=0.05)
        plt.savefig('%s/%i_%i.png' % (log_dir, epoch, batches_done), bbox_inches='tight')
        plt.close()


class ImagePlotSaveV2:
    def __init__(self, output_shape, input_shape):
        self.output_shape = output_shape
        self.input_shape = input_shape

    # 保存图片
    def sample_images(self, epoch, batches_done, log_dir, real_A, fake_A, real_B, fake_B):
        """Saves a generated sample from the test set"""

        real_B_r = real_B[0, 0, :, :]
        real_B_i = real_B[0, 1, :, :]
        fake_B_r = fake_B[0, 0, :, :]
        fake_B_i = fake_B[0, 1, :, :]
        real_A = real_A[0, 0, :, :]
        fake_A = fake_A[0, 0, :, :]

        # --------------------------2022.11.06 add ---------------------------
        real_B_r = real_B_r.cpu().squeeze().detach().numpy()
        real_B_i = real_B_i.cpu().squeeze().detach().numpy()
        fake_B_r = fake_B_r.cpu().squeeze().detach().numpy()
        fake_B_i = fake_B_i.cpu().squeeze().detach().numpy()
        real_A = real_A.cpu().squeeze().detach().numpy()
        fake_A = fake_A.cpu().squeeze().detach().numpy()
        plt.figure(figsize=(14, 14), dpi=300)
        x1 = plt.subplot(2, 3, 1)
        plt.imshow(real_A)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 0)
        x1.set_title('Mask')
        x2 = plt.subplot(2, 3, 2)
        plt.imshow(fake_B_r)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 1)
        x2.set_title('Generated real part of NF')
        x3 = plt.subplot(2, 3, 3)
        plt.imshow(fake_B_i)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x3.set_title('Generated imaginary part of NF')
        x4 = plt.subplot(2, 3, 4)
        plt.imshow(fake_A)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 0)
        x4.set_title('Generated_Mask')
        x5 = plt.subplot(2, 3, 5)
        plt.imshow(real_B_r)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 1)
        x5.set_title('real part of NF')
        x6 = plt.subplot(2, 3, 6)
        plt.imshow(real_B_i)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x6.set_title('imaginary part of NF')

        plt.subplots_adjust(wspace=0.4, hspace=0.05)
        plt.savefig('%s/%i_%i.png' % (log_dir, epoch, batches_done), bbox_inches='tight')
        plt.close()

    def plot_line(selft, epoch, loss_G, loss_D, valid_G_AB, train_G_AB, valid_G_BA, train_G_BA, valid_miou, train_miou, valid_acc, train_acc, valid_class_acc, train_class_acc, out_dir, mark='all'):
        """
        绘制训练和验证集的loss曲线
        """
        plt.figure(figsize=(28, 14), dpi=300)
        plt.subplot(221)
        plt.plot(epoch, loss_G, label='lossG', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, loss_D, label='lossD', color='r', marker='o', markerfacecolor='r', markersize=15)
        # # plt.plot(valid_z, valid_k, label='Valid', fontsize=100)
        #
        plt.xlabel('Epoch', fontsize=30)
        plt.tick_params(labelsize=30)
        # # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        # # plt.gca().axes.get_xaxis().set_visible(False)
        plt.title('loss', fontsize=30)

        plt.subplot(222)

        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, valid_G_AB, label='Valid', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, train_G_AB, label='Train', color='r', marker='o', markerfacecolor='r', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('G_AB', fontsize=30)

        plt.subplot(223)

        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, valid_G_BA, label='Valid', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, train_G_BA, label='Train', color='r', marker='o', markerfacecolor='r', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('G_BA', fontsize=30)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        plt.subplot(224)

        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, valid_miou, label='Valid_miou', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, train_miou, label='Train_miou', color='r', marker='o', markerfacecolor='r', markersize=15)
        plt.plot(epoch, valid_acc, label='Valid_acc', color='y', marker='o', markerfacecolor='y', markersize=15)
        plt.plot(epoch, train_acc, label='Train_acc', color='g', marker='o', markerfacecolor='g', markersize=15)
        plt.plot(epoch, valid_class_acc, label='Valid_class_acc', color='c', marker='o', markerfacecolor='c', markersize=15)
        plt.plot(epoch, train_class_acc, label='Train_class_acc', color='m', marker='o', markerfacecolor='m', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('G_BA效果', fontsize=30)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        plt.savefig(os.path.join(out_dir, mark + r'loss.png'))
        plt.close()

    # 绘制会议需要的训练曲线
    def plot_line_v2(selft, epoch, loss_G, loss_D, valid_G_AB, train_G_AB, valid_miou, train_miou ,out_dir, mark='all'):

        plt.figure(figsize=(28, 14), dpi=300)

        plt.subplot(131)
        plt.plot(epoch, loss_G, label='lossG', color='b', marker='o', markerfacecolor='b', markersize=10)
        plt.plot(epoch, loss_D, label='lossD', color='r', marker='o', markerfacecolor='r', markersize=10)
        plt.xlabel('Epoch', fontsize=30)
        plt.tick_params(labelsize=30)
        plt.legend(loc='best', prop={'size': 30})
        plt.title('loss', fontsize=30)

        plt.subplot(132)
        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, valid_G_AB, label='Valid', color='b', marker='o', markerfacecolor='b', markersize=10)
        plt.plot(epoch, train_G_AB, label='Train', color='r', marker='o', markerfacecolor='r', markersize=10)
        plt.tick_params(labelsize=30)
        plt.legend(loc='best', prop={'size': 30})
        plt.title('G_AB_loss', fontsize=30)


        plt.subplot(133)
        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, valid_miou, label='Valid_miou', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, train_miou, label='Train_miou', color='r', marker='o', markerfacecolor='r', markersize=10)
        plt.tick_params(labelsize=30)
        plt.legend(loc='best', prop={'size': 30})
        plt.title('G_BA_MIOU', fontsize=30)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.savefig(os.path.join(out_dir, mark + r'loss.png'))
        plt.close()

    def plot_line_test(selft, epoch, time_mean_AB, eval_mean_AB, eval_mean_AB_r, eval_mean_AB_i, time_mean_BA,
                       eval_mean_BA, out_dir):
        """
        绘制训练和验证集的loss曲线
        """
        plt.figure(figsize=(28, 14), dpi=300)
        plt.subplot(221)
        plt.plot(epoch, time_mean_AB, label='time_mean_AB', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, time_mean_BA, label='time_mean_BA', color='r', marker='o', markerfacecolor='r', markersize=15)

        #
        plt.xlabel('Epoch', fontsize=30)
        plt.tick_params(labelsize=30)
        plt.legend(loc='best', prop={'size': 30})
        plt.title('time', fontsize=30)

        plt.subplot(222)

        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, eval_mean_AB, label='G_AB_loss', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.plot(epoch, eval_mean_AB_r, label='G_BA_loss_r', color='r', marker='o', markerfacecolor='r', markersize=15)
        plt.plot(epoch, eval_mean_AB_i, label='G_BA_loss_i', color='y', marker='o', markerfacecolor='y', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('loss_AB', fontsize=30)

        plt.subplot(223)

        plt.xlabel('Epoch', fontsize=30)
        plt.plot(epoch, eval_mean_BA, label='G_BA_loss', color='b', marker='o', markerfacecolor='b', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('loss_BA', fontsize=30)




        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        plt.savefig(os.path.join(out_dir, r'test_result.png'))
        plt.close()

    def sample_images_v2(self, epoch, batches_done, log_dir, real_A, fake_A, real_B, fake_B,de_transforms_=None):
        """Saves a generated sample from the test set"""
        tf = transforms.Compose(de_transforms_)

        real_B_r = tf(real_B)[0, 0, :, :]
        real_B_i = tf(real_B)[0, 1, :, :]
        fake_B_r = tf(fake_B)[0, 0, :, :]
        fake_B_i = tf(fake_B)[0, 1, :, :]
        real_A = real_A[0, :, :, :].max(dim=0)[1].data
        fake_A = fake_A[0, :, :, :].max(dim=0)[1].data

        real_B_r = real_B_r.cpu().squeeze().detach().numpy()
        real_B_i = real_B_i.cpu().squeeze().detach().numpy()
        fake_B_r = fake_B_r.cpu().squeeze().detach().numpy()
        fake_B_i = fake_B_i.cpu().squeeze().detach().numpy()
        real_A = real_A.cpu().squeeze().detach().numpy()
        fake_A = fake_A.cpu().squeeze().detach().numpy()
        plt.figure(figsize=(14, 14), dpi=300)
        x1 = plt.subplot(2, 3, 1)
        plt.imshow(real_A)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(0, 1)
        x1.set_title('Mask')
        x2 = plt.subplot(2, 3, 2)
        plt.imshow(fake_B_r)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 1)
        x2.set_title('Generated real part of NF')
        x3 = plt.subplot(2, 3, 3)
        plt.imshow(fake_B_i)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x3.set_title('Generated imaginary part of NF')
        x4 = plt.subplot(2, 3, 4)
        plt.imshow(fake_A)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(0, 1)
        x4.set_title('Generated_Mask')
        x5 = plt.subplot(2, 3, 5)
        plt.imshow(real_B_r)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1, 1)
        x5.set_title('real part of NF')
        x6 = plt.subplot(2, 3, 6)
        plt.imshow(real_B_i)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x6.set_title('imaginary part of NF')

        plt.subplots_adjust(wspace=0.4, hspace=0.05)
        plt.savefig('%s/%i_%i.png' % (log_dir, epoch, batches_done), bbox_inches='tight')
        plt.close()
