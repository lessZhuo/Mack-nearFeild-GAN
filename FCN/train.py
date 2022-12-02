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
        mask = mask.max(dim=1)[1].data
        label = label.max(dim=1)[1].data

    mask = mask[0, 0, :, :].cpu().detach().numpy()
    nf_r = near_field[0, 0, :, :].cpu().detach().numpy()
    nf_i = near_field[0, 1, :, :].cpu().detach().numpy()
    label_1 = label[0, 0, :, :].cpu().detach().numpy()

    if bw is False:
        label_2 = label[0, 1, :, :].cpu().detach().numpy()

    plt.figure(figsize=(14, 14), dpi=300)

    x1 = plt.subplot(2, 3, 1)
    plt.imshow(mask)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x1.set_title('mask')

    x2 = plt.subplot(2, 3, 2)
    plt.imshow(nf_r)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x2.set_title('NF_r')

    x3 = plt.subplot(2, 3, 3)
    plt.imshow(nf_i)
    plt.colorbar(fraction=0.05, pad=0.05)
    # plt.clim(-1.001, -0.95)
    x3.set_title('NF_i')

    if bw:
        x4 = plt.subplot(2, 3, 4)
        plt.imshow(label_1)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x4.set_title('label_mask')
    else:
        x5 = plt.subplot(2, 3, 5)
        plt.imshow(label_1)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x5.set_title('label_NF_i')

        x6 = plt.subplot(2, 3, 6)
        plt.imshow(label_2)
        plt.colorbar(fraction=0.05, pad=0.05)
        # plt.clim(-1.001, -0.95)
        x6.set_title('label_NF_i')

    plt.subplots_adjust(wspace=0.4, hspace=0.05)
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
