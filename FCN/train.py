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

class ModelTrainer(object):
    @staticmethod
    def train(model, train_data, criterion, optimizer, epoch, device):
        best = [0]
        net = model.train()

        train_loss = []

        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['A'].to(device))
            img_label = Variable(sample['B'].to(device))
            # 训练
            out = net(img_data)
            # out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        return np.mean(train_loss)

    @staticmethod
    def evaluate(model, val_data, criterion, epoch, device, log_dir):
        net = model.eval()
        eval_loss = []

        prec_time = datetime.now()
        for j, sample in enumerate(val_data):
            valImg = Variable(sample['A'].to(device))
            valLabel = Variable(sample['B'].to(device))

            out = net(valImg)

            loss = criterion(out, valLabel)
            eval_loss.append(loss.item())
            if j % 100 == 0:
                out_mask = valImg[0, 0, :, :].cpu().detach().numpy()
                label_r = valLabel[0, 0, :, :].cpu().detach().numpy()
                label_i = valLabel[0, 1, :, :].cpu().detach().numpy()
                out_label_r = out[0, 0, :, :].cpu().detach().numpy()
                out_label_i = out[0, 1, :, :].cpu().detach().numpy()

                plt.figure(figsize=(14, 14), dpi=300)

                x1 = plt.subplot(2, 3, 1)
                plt.imshow(out_mask)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x1.set_title('mask')

                x2 = plt.subplot(2, 3, 2)
                plt.imshow(out_label_r)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x2.set_title('out_NF_r')

                x3 = plt.subplot(2, 3, 3)
                plt.imshow(out_label_i)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x3.set_title('out_NF_i')

                x3 = plt.subplot(2, 3, 5)
                plt.imshow(label_i)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x3.set_title('NF_i')

                x3 = plt.subplot(2, 3, 6)
                plt.imshow(label_r)
                plt.colorbar(fraction=0.05, pad=0.05)
                # plt.clim(-1.001, -0.95)
                x3.set_title('NF_i')




                plt.subplots_adjust(wspace=0.4, hspace=0.05)
                plt.savefig('%s/%i_%i.png' % (log_dir, epoch, j), bbox_inches='tight')
                plt.close()

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
