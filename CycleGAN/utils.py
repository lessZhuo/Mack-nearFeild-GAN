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
    def sample_images(self, batches_done, log_dir, real_A, fake_A, real_B, fake_B):
        """Saves a generated sample from the test set"""

        real_B_r = real_B[0, 0, :, :]
        real_B_i = real_B[0, 1, :, :]
        fake_B_r = fake_B[0, 0, :, :]
        fake_B_i = fake_B[0, 1, :, :]
        real_A = real_A[0, 0, :, :]
        fake_A = fake_A[0, 0, :, :]

        # # Arange images along x-axi
        # r_A = make_grid(real_A, nrow=1, normalize=True)
        # r_B_r = make_grid(real_B_r, nrow=1, normalize=True)
        # r_B_i = make_grid(real_B_i, nrow=1, normalize=True)
        #
        # f_A = make_grid(fake_A, nrow=1, normalize=True)
        # f_B_r = make_grid(fake_B_r, nrow=1, normalize=True)
        # f_B_i = make_grid(fake_B_i, nrow=1, normalize=True)
        # # Arange images along y-axis

        # image_grid = torch.cat((r_A, f_B_r, f_B_i, r_B_r, r_B_i, f_A), 1)

        # --------------------------2022.11.06 add ---------------------------
        real_B_r = real_B_r.cpu().squeeze().detach().numpy()
        real_B_i = real_B_i.cpu().squeeze().detach().numpy()
        fake_B_r = fake_B_r.cpu().squeeze().detach().numpy()
        fake_B_i = fake_B_i.cpu().squeeze().detach().numpy()
        real_A = real_A.cpu().squeeze().detach().numpy()
        fake_A = fake_A.cpu().squeeze().detach().numpy()

        x1 = plt.subplot(2, 3, 1)
        plt.imshow(real_A)
        x1.set_title('real_A')
        x2 = plt.subplot(2, 3, 2)
        plt.imshow(fake_B_r)
        x2.set_title('fake_B_r')
        x3 = plt.subplot(2, 3, 3)
        plt.imshow(fake_B_i)
        x3.set_title('fake_B_i')
        x4 = plt.subplot(2, 3, 4)
        plt.imshow(fake_A)
        x4.set_title('fake_A')
        x5 = plt.subplot(2, 3, 5)
        plt.imshow(real_B_r)
        x5.set_title('real_B_r')
        x6 = plt.subplot(2, 3, 6)
        plt.imshow(real_B_i)
        x6.set_title('real_B_i')

        # image_grid = image_grid.cpu()  # .transpose(0, 3, 1, 2)
        # image_grid = image_grid[0, :, :].detach().numpy()  # [0, 0, :, :]

        # y 轴不可见
        # plt.gca().axes.get_yaxis().set_visible(False)
        # fig, ax = plt.subplots()
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 输出图像#边框设置
        # plt.imshow(image_grid)
        # plt.axis('off')
        # plt.gca().xaxis.set_ticks_position('top')
        # plt.tick_params(labelsize=30)
        # plt.title('yy_real')
        # plt.show()
        # ax = plt.subplot(111)
        # ax.invert_yaxis()  # y轴反向
        # ax.set_title('xx_real', fontsize=20)

        # plt.colorbar()  # fraction=0.05, pad=0.05
        # cb = plt.colorbar()  # fraction=0.05, pad=0.05
        # cb.ax.tick_params(labelsize=30)

        plt.savefig('%s/%i.png' % (log_dir, batches_done), bbox_inches='tight')
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
        plt.plot(train_x, train_y, label='lossG', color='b', marker='o', markerfacecolor='b', markersize=10)
        plt.plot(valid_x, valid_y, label='lossD', color='r', marker='o', markerfacecolor='r', markersize=10)
        # # plt.plot(valid_z, valid_k, label='Valid', fontsize=100)
        #
        # plt.ylabel(str(mode), fontsize=100)
        # # plt.xlabel('Epoch', fontsize=100)
        plt.tick_params(labelsize=30)
        # # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        # # plt.gca().axes.get_xaxis().set_visible(False)
        plt.title('yy_real loss', fontsize=30)

        plt.subplot(212)

        # plt.ylabel(str('Valid Loss'))
        plt.xlabel('Epoch', fontsize=30)
        # plt.title('Valid Loss', fontsize=10)
        plt.plot(valid_z, valid_k, label='Valid', color='b', marker='o', markerfacecolor='b', markersize=10)
        plt.plot(train_m, train_n, label='Train', color='r', marker='o', markerfacecolor='r', markersize=10)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        # plt.gca().axes.get_xaxis().set_visible(True)
        # plt.figure(figsize=(14, 14), dpi=300)
        plt.savefig(os.path.join(out_dir, mode + '.tiff'))
        plt.close()
