import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from text.dataset import LoadDataset
from text import FCN
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'

device = t.device('cpu')

# device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
BASE_DIR = r'E:\Datasets\近场数据\RECT_ARRAY_MASK'
BASE_DIR1 = r"E:\Datasets\近场数据\RECT_ARRAY_NF_x0_y0"
crop_size = (300, 300)
Epoch = 2
BATCH_SIZE = 20

Load_train = LoadDataset([BASE_DIR, BASE_DIR1], crop_size)
Load_val = LoadDataset([BASE_DIR, BASE_DIR1], crop_size)

train_data = DataLoader(Load_train, BATCH_SIZE, shuffle=True, num_workers=1)
val_data = DataLoader(Load_val, BATCH_SIZE, shuffle=True, num_workers=1)

fcn = FCN.FCN()
fcn = fcn.to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)


class ModelTrainer(object):

    @staticmethod
    def train(model, train_data, criterion, optimizer, epoch, device):
        best = [0]
        net = model.train()

        train_loss = []

        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['mask'].to(device))
            img_label = Variable(sample['label'].to(device))
            # 训练
            out = net(img_data)
            # out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        #
        # out_mask=out[0,0,100:200,100:200].detach().numpy()
        # out_label=img_label[0,0,100:200,100:200].detach().numpy()
        # plt.imshow(out_mask)
        # plt.imsave('mask_%i'%(epoch+i),out_mask,format='png')
        # plt.imshow(out_label)
        # plt.imsave('label_%i'%(epoch+i),out_label,format='png')

        return np.mean(train_loss)

    @staticmethod
    def evaluate(model, val_data, criterion, epoch, device, log_dir):
        net = model.eval()
        eval_loss = []
        eval_acc = 0
        eval_miou = 0
        eval_class_acc = 0

        prec_time = datetime.now()
        for j, sample in enumerate(val_data):
            valImg = Variable(sample['mask'].to(device))
            valLabel = Variable(sample['label'].to(device))

            out = net(valImg)

            loss = criterion(out, valLabel)
            eval_loss.append(loss.item())
            if epoch > 40:
                out_mask = out[0, 0, :, :].cpu().detach().numpy()
                out_label = valLabel[0, 0, :, :].cpu().detach().numpy()
                plt.imshow(out_mask)
                # plt.imsave('mask_%i_%i.png'%(epoch,j),out_mask,format='png')
                plt.colorbar(fraction=0.05, pad=0.05)
                plt.axis('off')
                plt.savefig(os.path.join(log_dir, 'mask_%i_%i.png' % (epoch, j)))
                plt.close()
                plt.imshow(out_label)
                # plt.imsave('label_%i_%i.png'%(epoch,j),out_label,format='png')
                plt.colorbar(fraction=0.05, pad=0.05)
                plt.axis('off')
                plt.savefig(os.path.join(log_dir, 'label_%i_%i.png' % (epoch, j)))
                plt.close()

        return np.mean(eval_loss)


if __name__ == "__main__":
    ModelTrainer.train(fcn,train_data,criterion,optimizer,Epoch,device)
