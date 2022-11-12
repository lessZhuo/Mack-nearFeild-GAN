import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
import itertools


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data[0]
            else:
                self.losses[loss_name] += losses[loss_name].data[0]

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                               opts={'title': image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
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


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


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

        x1 = plt.subplot(2, 3, 1)
        plt.imshow(real_A)
        plt.colorbar()
        plt.clim(-1, 0)
        x1.set_title('real_A')
        x2 = plt.subplot(2, 3, 2)
        plt.imshow(fake_B_r)
        plt.colorbar()
        plt.clim(-1, 1)
        x2.set_title('fake_B_r')
        x3 = plt.subplot(2, 3, 3)
        plt.imshow(fake_B_i)
        plt.colorbar()
        plt.clim(-1.001, -0.95)
        x3.set_title('fake_B_i')
        x4 = plt.subplot(2, 3, 4)
        plt.imshow(fake_A)
        plt.colorbar()
        plt.clim(-1, 0)
        x4.set_title('fake_A')
        x5 = plt.subplot(2, 3, 5)
        plt.imshow(real_B_r)
        plt.colorbar()
        plt.clim(-1, 1)
        x5.set_title('real_B_r')
        x6 = plt.subplot(2, 3, 6)
        plt.imshow(real_B_i)
        plt.colorbar()
        plt.clim(-1.001, -0.95)
        x6.set_title('real_B_i')

        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        plt.savefig('%s/%i_%i.png' % (log_dir,epoch, batches_done), bbox_inches='tight')
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

    def plot_line_test(selft, epoch, time_mean_AB, eval_mean_AB, eval_mean_AB_r, eval_mean_AB_i, time_mean_BA, eval_mean_BA, eval_mean_BA_r, eval_mean_BA_i, out_dir):
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
        plt.plot(epoch, eval_mean_BA_r, label='G_BA_loss_r', color='r', marker='o', markerfacecolor='r', markersize=15)
        plt.plot(epoch, eval_mean_BA_i, label='G_BA_loss_i', color='y', marker='o', markerfacecolor='y', markersize=15)
        plt.tick_params(labelsize=30)
        # location = 'upper right' if mode == 'loss' else 'upper left'
        plt.legend(loc='best', prop={'size': 30})
        plt.title('loss_BA', fontsize=30)

        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        plt.savefig(os.path.join(out_dir, r'test_result.png'))
        plt.close()
