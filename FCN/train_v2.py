import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import LoadDataset
import FCN
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'


def save_img(mask, near_field, label, epoch, num, log_dir, bw=False):
    if bw:
        label = label.max(dim=1)[1].data
        mask = mask[0, :, :].cpu().detach().numpy()
        label_1 = label[0, :, :].cpu().detach().numpy()
    else:
        mask = mask[0, 0, :, :].cpu().detach().numpy()
        label_1 = label[0, 0, :, :].cpu().detach().numpy()
        label_2 = label[0, 1, :, :].cpu().detach().numpy()
        label_3 = label[0, 2, :, :].cpu().detach().numpy()
        label_4 = label[0, 3, :, :].cpu().detach().numpy()
        label_5 = label[0, 4, :, :].cpu().detach().numpy()
        label_6 = label[0, 5, :, :].cpu().detach().numpy()
        label_7 = label[0, 6, :, :].cpu().detach().numpy()
        label_8 = label[0, 7, :, :].cpu().detach().numpy()

    nf_xx_r = near_field[0, 0, :, :].cpu().detach().numpy()
    nf_xx_i = near_field[0, 1, :, :].cpu().detach().numpy()
    nf_yy_r = near_field[0, 2, :, :].cpu().detach().numpy()
    nf_yy_i = near_field[0, 3, :, :].cpu().detach().numpy()
    nf_xy_r = near_field[0, 4, :, :].cpu().detach().numpy()
    nf_xy_i = near_field[0, 5, :, :].cpu().detach().numpy()
    nf_yx_r = near_field[0, 6, :, :].cpu().detach().numpy()
    nf_yx_i = near_field[0, 7, :, :].cpu().detach().numpy()

    plt.figure(figsize=(90, 10), dpi=300)

    x1 = plt.subplot(2, 9, 1)
    plt.imshow(mask)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x1.set_title('mask')

    x2 = plt.subplot(2, 9, 2)
    plt.imshow(nf_xx_r)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x2.set_title('NF_xx_r')

    x3 = plt.subplot(2, 9, 3)
    plt.imshow(nf_xx_i)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x3.set_title('NF_xx_i')

    x4 = plt.subplot(2, 9, 4)
    plt.imshow(nf_yy_r)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x4.set_title('NF_yy_r')

    x5 = plt.subplot(2, 9, 5)
    plt.imshow(nf_yy_i)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x5.set_title('NF_yy_i')

    x6 = plt.subplot(2, 9, 6)
    plt.imshow(nf_xy_r)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x6.set_title('NF_xy_r')

    x7 = plt.subplot(2, 9, 7)
    plt.imshow(nf_xy_i)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x7.set_title('NF_xy_i')

    x8 = plt.subplot(2, 9, 8)
    plt.imshow(nf_xy_r)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x8.set_title('NF_yx_r')

    x9 = plt.subplot(2, 9, 9)
    plt.imshow(nf_xy_i)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x9.set_title('NF_yx_i')

    if bw:
        x10 = plt.subplot(2, 9, 10)
        plt.imshow(label_1)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x10.set_title('out_mask')
    else:
        x11 = plt.subplot(2, 9, 11)
        plt.imshow(label_1)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x11.set_title('out_NF_xx_r')

        x12 = plt.subplot(2, 9, 12)
        plt.imshow(label_2)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x12.set_title('out_NF_xx_i')

        x13 = plt.subplot(2, 9, 13)
        plt.imshow(label_3)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x13.set_title('out_NF_yy_r')

        x14 = plt.subplot(2, 9, 14)
        plt.imshow(label_4)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x14.set_title('out_NF_yy_i')

        x15 = plt.subplot(2, 9, 15)
        plt.imshow(label_5)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x15.set_title('out_NF_xy_r')

        x16 = plt.subplot(2, 9, 16)
        plt.imshow(label_6)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x16.set_title('out_NF_xy_i')

        x17 = plt.subplot(2, 9, 17)
        plt.imshow(label_7)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x17.set_title('out_NF_yx_r')

        x18 = plt.subplot(2, 9, 18)
        plt.imshow(label_8)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x18.set_title('out_NF_yx_i')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig('%s/%i_%i.png' % (log_dir, epoch, num), bbox_inches='tight')
    plt.close()


class ModelTrainer(object):
    @staticmethod
    def train(model, train_data, criterion, optimizer, epoch, device, bw=False):
        best = [0]
        net = model.train()

        train_loss = []

        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            if bw:
                img_data = Variable(sample['B'].to(device))
                img_label = Variable(sample['C'].to(device))
            else:
                img_data = Variable(sample['A'].to(device))
                img_label = Variable(sample['B'].to(device))
            # 训练
            out = net(img_data)
            if bw:
                out = F.log_softmax(out, dim=1)
            # print(out.shape)
            # print(img_label.shape)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        return np.mean(train_loss)

    @staticmethod
    def evaluate(model, val_data, criterion, epoch, device, log_dir, bw=False):
        net = model.eval()
        eval_loss = []

        prec_time = datetime.now()
        for j, sample in enumerate(val_data):
            if bw:
                valImg = Variable(sample['B'].to(device))
                valLabel = Variable(sample['C'].to(device))
            else:
                valImg = Variable(sample['A'].to(device))
                valLabel = Variable(sample['B'].to(device))

            out = net(valImg)

            if bw:
                out = F.log_softmax(out, dim=1)

            loss = criterion(out, valLabel)
            eval_loss.append(loss.item())

            if j % 100 == 0:
                if bw:
                    save_img(valLabel, valImg, out, epoch, j, log_dir, bw)
                else:
                    save_img(valImg, valLabel, out, epoch, j, log_dir, bw)

        return np.mean(eval_loss)


if __name__ == "__main__":
    device = t.device('cpu')

    # device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    BASE_DIR = r'../dataset/fcn'
    crop_size = (300, 300)
    Epoch = 2
    BATCH_SIZE = 20

    transforms_ = [
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize([1], [1]),
    ]

    Load_train = LoadDataset(BASE_DIR, transforms_=transforms_, mode="train", combine=False, direction="x", part="real")
    Load_val = LoadDataset(BASE_DIR, transforms_=transforms_, mode="text", combine=False, direction="x", part="real")

    train_data = DataLoader(Load_train, BATCH_SIZE, shuffle=True, num_workers=1)
    val_data = DataLoader(Load_val, BATCH_SIZE, shuffle=True, num_workers=1)

    fcn = FCN.FCN()
    fcn = fcn.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(fcn.parameters(), lr=1e-4)
    ModelTrainer.train(fcn, train_data, criterion, optimizer, Epoch, device)
