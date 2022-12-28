import argparse
import torch.nn.functional as F
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
from models_sa import *
from datasets_v2 import *
from utils import *
import torch
from evalution_segmentaion import eval_semantic_segmentation

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
# ---------------------------------参数设置---------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")  # 128
parser.add_argument("--img_width", type=int, default=128, help="size of image width")  # 128
parser.add_argument("--sample_interval", type=int, default=250, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=5, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--input_channels", type=int, default=2, help="number of input channels")
parser.add_argument("--output_channels", type=int, default=8, help="number of image channels")
parser.add_argument("--proportion", type=float, default=0.4, help="proportion of A to B loss in total loss ")
opt = parser.parse_args()

# Create sample and checkpoint directories
# -------------------------------设置保存的数据和结果模型------------------------------------------------
os.makedirs("../result/cycleGan/images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("../result/cycleGan/saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_NLL = torch.nn.NLLLoss()

cuda = torch.cuda.is_available()
Epoch = opt.n_epochs
# ---------------------------关系到模型参数的设置------------------------------
# 如果要设置合并起来计算 输出的通道必须设置为2，则为实部和虚部一起训练
input_shape = (opt.input_channels, opt.img_height, opt.img_width)
output_shape = (opt.output_channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator --- 网络模型
G_AB = GeneratorResNet(input_shape, output_shape, opt.n_residual_blocks, fw=True)
G_BA = GeneratorResNet(output_shape, input_shape, opt.n_residual_blocks, fw=False)
# ----------------测试反向生成为2分类分割问题------------------------
# G_BA = GeneratorResNet(output_shape, output_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(output_shape)

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
    criterion_Vail.cuda()
    criterion_NLL.cuda()

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

# save image and plot loss line
image_save_plot = ImagePlotSaveV3(output_shape, input_shape)

# transformations
# transforms_ = [
#     transforms.Normalize(mean=[0.193, 0.195], std=[0.927, 1.378])
# ]

# transforms_ = [
#     # transforms.Normalize(mean=[0.0062, 0.0048], std=[1.0016, 1.0003])
#     transforms.Normalize(mean=[0.193, 0.195, 0.193, 0.195, 0.193, 0.195, 0.193, 0.195],
#                          std=[0.927, 1.378, 0.927, 1.378, 0.927, 1.378, 0.927, 1.378])
# ]

transforms_ = [
    # transforms.Normalize(mean=[0.0062, 0.0048], std=[1.0016, 1.0003])
    transforms.Normalize(mean=[0.193, 0.195, 0.193, 0.195, -1.5679, 0.208, 0.6426, -0.6321] ,
                         std=[0.927, 1.378, 0.927, 1.378, 10, 2, 10, 2])
]

# de_transforms_ = [
#     # transforms.ToTensor(),
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     transforms.Normalize(mean=[-0.2082, -0.1415], std=[1.0787, 0.7257]),
# ]

de_transforms_ = [
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize(
        mean=[-0.2082, -0.1415, -0.2082, -0.1415, -1.5679 /10 , 0.208 / 2, 0.6426 / 10, -0.6321 / 2],
        std=[1.0787, 0.7257, 1.0787, 0.7257, 0.1, 0.5, 0.1, 0.5]),
]
# transforms.Normalize(mean=[-0.2082, -0.1415,-0.2082, -0.1415,-0.0000000016426, -0.0000000051376,-0.0000000017808, 0.000000005534],
#                          std=[1.0787, 0.7257,1.0787, 0.7257,7246, 24780,15983, 65190]),


# Training data loader
dataloader = DataLoader(
    MaskNfDatasetV2("../datasets/crop_128/final", transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    MaskNfDatasetV2("../datasets/crop_128/final", transforms_=transforms_, mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

# proportion of loss
proportion = opt.proportion

# now_time = datetime.datetime.now()
# time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
# log_dir = os.path.join("F:/JZJ/hello pytorch/My_Code", "results", time_str)


# ----------
#  Training
# ----------

if __name__ == '__main__':
    print(device)
    loss_rec = {"loss_D": [], "loss_G": [], "loss_G_AB_valid": [], "loss_G_AB_train": [], "loss_G_BA_valid": [],
                "loss_G_BA_train": [], "G_BA_Miou_train": [], "G_BA_Miou_valid": [], "G_BA_acc_valid": [],
                "G_BA_acc_train": [], "G_BA_class_acc_valid": [], "G_BA_class_acc_train": []}

    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join("../results/", "cycleGan", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    prev_time = time.time()
    # i, batch = next(enumerate(dataloader))
    best_acc, best_epoch = 0, 0
    best_loss = 0.01
    best_miou = 0
    lambda_cyc_A = 10

    for epoch in range(opt.epoch, opt.n_epochs):
        D_loss = []
        G_loss = []
        train_loss_AB = []
        train_loss_BA = []
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        torch.cuda.empty_cache()
        if epoch % 10 == 0 and epoch != 0:
            lambda_cyc_A *= 1.5

        for i, batch in enumerate(dataloader):
            # 清除缓存
            torch.cuda.empty_cache()
            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            real_A_label = Variable(batch["C"].type(Tensor).long())

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # ----------需要修改验证模型的格式 ----------
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = loss_GAN_AB * proportion + loss_GAN_BA * (1 - proportion)

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_NLL(recov_A, real_A_label)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = loss_cycle_B * proportion + loss_cycle_A * (1 - proportion) * lambda_cyc_A

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle  # + opt.lambda_id * loss_identity
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

            loss_D = loss_D_B * proportion + loss_D_A * (1 - proportion)
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
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f,nll_loss: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_cycle_A.item(),
                    time_left,
                )
            )

            # --------------
            #  vail train Progress
            # --------------

            net_G_AB = G_AB.eval()
            net_G_BA = G_BA.eval()

            fake_B = net_G_AB(real_A)
            fake_A = net_G_BA(real_B)

            loss_AB = criterion_Vail(fake_B, real_B)
            train_loss_AB.append(loss_AB.item())

            loss_BA = criterion_NLL(fake_A, real_A_label)
            train_loss_BA.append(loss_BA.item())

            pre_label = fake_A.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = real_A_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                image_save_plot.sample_images_v2(epoch, batches_done, log_dir, real_A=real_A, real_B=real_B,
                                                 fake_A=fake_A,
                                                 fake_B=fake_B, de_transforms_=de_transforms_)

        train_mean_AB = np.mean(train_loss_AB)
        train_mean_BA = np.mean(train_loss_BA)
        train_mean_miou = train_miou / len(dataloader)
        train_mean_acc = train_acc / len(dataloader)
        train_mean_class_acc = train_class_acc / len(dataloader)

        print(
            "Epoch[{:0>3}/{:0>3}]  train_AB_loss:{:.6f}  train_BA_loss:{:.6f}  Miou :{:.6f}  Train_acc:{:.6f}  Train_class_acc:{:}  ".format(
                epoch, Epoch, train_mean_AB, train_mean_BA, train_mean_miou, train_mean_acc, train_mean_class_acc))

        # --------------
        #  vail Progress
        # --------------

        net_G_AB_v = G_AB.eval()
        net_G_BA_v = G_BA.eval()
        eval_loss_G_AB = []
        eval_loss_G_BA = []
        eval_acc = 0
        eval_miou = 0
        eval_class_acc = 0
        # 验证用论文的函数验证 MSE验证
        for j, sample in enumerate(val_dataloader):
            real_A = Variable(sample['A'].to(device).float())
            real_B = Variable(sample['B'].to(device))
            real_A_label = Variable(sample['C'].to(device))

            fake_B = net_G_AB_v(real_A)
            fake_A = net_G_BA_v(real_B)

            loss_G_AB = criterion_Vail(fake_B, real_B)
            eval_loss_G_AB.append(loss_G_AB.item())

            # 评估 mask 的参数 采用miou评估
            pre_label = fake_A.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = real_A_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            eval_acc += eval_metrix['mean_class_accuracy']
            eval_miou += eval_metrix['miou']
            eval_class_acc += eval_metrix['class_accuracy']

            loss_G_BA = criterion_NLL(fake_A, real_A_label)
            eval_loss_G_BA.append(loss_G_BA.item())

        G_mean = np.mean(G_loss)
        D_mean = np.mean(D_loss)
        eval_mean_AB = np.mean(eval_loss_G_AB)
        eval_mean_BA = np.mean(eval_loss_G_BA)
        eval_mean_miou = eval_miou / len(val_dataloader)
        eval_mean_acc = eval_acc / len(val_dataloader)
        eval_mean_class_acc = eval_class_acc / len(val_dataloader)

        print(
            "Epoch[{:0>3}/{:0>3}]  loss_G:{:.6f} loss_D:{:.6f} loss_G_AB_valid:{:.9f} loss_G_BA_valid:{:.9f} train_G_AB_loss:{:.9f} train_G_BA_loss:{:.9f}".format(
                epoch, Epoch, G_mean, D_mean, eval_mean_AB, eval_mean_BA, train_mean_AB, train_mean_BA))
        print(
            "train_Miou:{:.9f} valid_Miou:{:.9f} train_acc:{:.9f} valid_acc:{:.9f} train_class_acc:{:} valid_class_acc:{:}".format(
                train_mean_miou, eval_mean_miou, train_mean_acc, eval_mean_acc, train_mean_class_acc,
                eval_mean_class_acc))

        # 绘图
        loss_rec["loss_G"].append(G_mean), loss_rec["loss_D"].append(D_mean),
        loss_rec["loss_G_AB_valid"].append(eval_mean_AB), loss_rec["loss_G_BA_valid"].append(eval_mean_BA), \
        loss_rec["loss_G_AB_train"].append(train_mean_AB), loss_rec["loss_G_BA_train"].append(train_mean_BA)
        loss_rec["G_BA_Miou_valid"].append(eval_mean_miou), loss_rec["G_BA_Miou_train"].append(train_mean_miou)
        loss_rec["G_BA_acc_valid"].append(eval_mean_acc), loss_rec["G_BA_acc_train"].append(train_mean_acc)
        loss_rec["G_BA_class_acc_valid"].append(eval_mean_class_acc), loss_rec["G_BA_class_acc_train"].append(
            train_mean_class_acc)

        plt_x = np.arange(1, epoch + 2)
        image_save_plot.plot_line(plt_x, loss_rec["loss_G"], loss_rec["loss_D"], loss_rec["loss_G_AB_valid"],
                                  loss_rec["loss_G_AB_train"], loss_rec["loss_G_BA_valid"], loss_rec["loss_G_BA_train"],
                                  loss_rec["G_BA_Miou_valid"], loss_rec["G_BA_Miou_train"],
                                  loss_rec["G_BA_acc_valid"], loss_rec["G_BA_acc_train"],
                                  loss_rec["G_BA_class_acc_valid"], loss_rec["G_BA_class_acc_train"],
                                  out_dir=log_dir)
        image_save_plot.plot_line_v2(plt_x, loss_rec["loss_G"], loss_rec["loss_D"],
                                     loss_rec["loss_G_AB_valid"], loss_rec["loss_G_AB_train"],
                                     loss_rec["G_BA_Miou_valid"], loss_rec["G_BA_Miou_train"],
                                     out_dir=log_dir, mark='all_v2')
        # ------------------------------------------temp-------------------------------------
        if epoch > 5:
            image_save_plot.plot_line(plt_x[5:], loss_rec["loss_G"][5:], loss_rec["loss_D"][5:],
                                      loss_rec["loss_G_BA_valid"][5:],
                                      loss_rec["loss_G_AB_train"][5:], loss_rec["loss_G_BA_valid"][5:],
                                      loss_rec["loss_G_BA_train"][5:],
                                      loss_rec["G_BA_Miou_valid"][5:], loss_rec["G_BA_Miou_train"][5:],
                                      loss_rec["G_BA_acc_valid"][5:], loss_rec["G_BA_acc_train"][5:],
                                      loss_rec["G_BA_class_acc_valid"][5:], loss_rec["G_BA_class_acc_train"][5:],
                                      out_dir=log_dir, mark='temp')
            image_save_plot.plot_line_v2(plt_x[5:], loss_rec["loss_G"][5:], loss_rec["loss_D"][5:],
                                         loss_rec["loss_G_BA_valid"][5:], loss_rec["loss_G_AB_train"][5:],
                                         loss_rec["G_BA_Miou_valid"][5:], loss_rec["G_BA_Miou_train"][5:],
                                         out_dir=log_dir, mark='temp_v2')

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if eval_mean_AB < max(best_loss, 0.01) or eval_mean_miou > min(best_miou, 0.8):
            best_loss = min(eval_mean_AB, best_loss)
            best_miou = max(eval_mean_miou, best_miou)
            best_epoch = epoch

            torch.save(G_AB.state_dict(), '%s/G_AB_%d.pth' % (log_dir, epoch))
            torch.save(G_BA.state_dict(), '%s/G_BA_%d.pth' % (log_dir, epoch))

    print(
        " done ~~~~ {}, best acc: {} in :{} epochs. ".format(
            datetime.datetime.strftime(datetime.datetime.now(), '%m-%d_%H-%M'),
            best_loss, best_epoch))
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
