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
from models import *
from datasets import *
from utils import *
import torch
from evalution_segmentaion import eval_semantic_segmentation

# ---------------------------------参数设置---------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")  # 128
parser.add_argument("--img_width", type=int, default=256, help="size of image width")  # 128
parser.add_argument("--sample_interval", type=int, default=250, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=5, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--input_channels", type=int, default=2, help="number of input channels")
parser.add_argument("--output_channels", type=int, default=2, help="number of image channels")
parser.add_argument("--proportion", type=float, default=0.4, help="proportion of A to B loss in total loss ")
opt = parser.parse_args()

# Create sample and checkpoint directories
# -------------------------------设置保存的数据和结果模型------------------------------------------------
os.makedirs("../result/cycleGan/images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("../result/cycleGan/saved_models/%s" % opt.dataset_name, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Losses

criterion_NLL = torch.nn.NLLLoss()
criterion_Vail = torch.nn.MSELoss().to(device)

cuda = torch.cuda.is_available()
Epoch = opt.n_epochs
# ---------------------------关系到模型参数的设置------------------------------
# 如果要设置合并起来计算 输出的通道必须设置为2，则为实部和虚部一起训练
input_shape = (opt.input_channels, opt.img_height, opt.img_width)
output_shape = (opt.output_channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator --- 网络模型
G_AB = GeneratorResNet(input_shape, output_shape, opt.n_residual_blocks, fw=True)
G_BA = GeneratorResNet(output_shape, input_shape, opt.n_residual_blocks, fw=False)



if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    criterion_Vail.cuda()
    criterion_NLL.cuda()

# ---------------------------------设置加载的参数---------------------------------------------
G_AB.load_state_dict(torch.load(r"F:\less\results\cycleGan\12-03_22-35\G_AB_24.pth"))
G_BA.load_state_dict(torch.load(r"F:\less\results\cycleGan\12-03_22-35\G_BA_24.pth"))

# save image and plot loss line
image_save_plot = ImagePlotSaveV2(output_shape, input_shape)

# transformations
transforms_ = [
    transforms.Normalize(mean=[0.193, 0.195], std=[0.927, 1.378])
]

de_transforms_ = [
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize(mean=[-0.2082, -0.1415], std=[1.0787, 0.7257]),
]

# Training data loader
dataloader = DataLoader(
    MaskNfDatasetV2("../datasets/crop_256/new", transforms_=transforms_, mode="test", combine=True, direction="x"),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# proportion of loss
proportion = opt.proportion

# ----------
#  Training
# ----------

if __name__ == '__main__':
    print(device)
    loss_rec = {"loss_G_AB": [], "loss_G_BA": [],
                "G_BA_Miou": [], "G_BA_acc": []}

    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join("../results/", "cycleGan/test", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    prev_time = time.time()
    # i, batch = next(enumerate(dataloader))
    best_acc, best_epoch = 0, 0
    best_loss = 0.01
    best_miou = 0
    lambda_cyc_A = 20

    for epoch in range(opt.epoch, opt.n_epochs):
        D_loss = []
        G_loss = []
        eval_loss_AB = []
        eval_loss_BA = []
        times_AB = []
        times_BA = []
        eval_loss_AB_r = []
        eval_loss_AB_i = []
        acc = 0
        miou = 0
        class_acc = 0
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

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done

            net_G_AB = G_AB.eval()
            net_G_BA = G_BA.eval()

            prev_time = time.time()
            fake_B = net_G_AB(real_A)
            curr_time = time.time()
            time_AB = curr_time - prev_time
            times_AB.append(time_AB)
            loss = criterion_Vail(fake_B, real_B)
            loss_r = criterion_Vail(fake_B[:, 0, :, :], real_B[:, 0, :, :])
            loss_i = criterion_Vail(fake_B[:, 1, :, :], real_B[:, 1, :, :])
            loss_AB = criterion_Vail(fake_B, real_B)
            eval_loss_AB.append(loss_AB.item())
            eval_loss_AB_r.append(loss_r.item())
            eval_loss_AB_i.append(loss_i.item())

            prev_time = time.time()
            fake_A = net_G_BA(real_B)
            curr_time = time.time()
            time_BA = curr_time - prev_time
            times_BA.append(time_BA)

            loss_BA = criterion_NLL(fake_A, real_A_label)
            eval_loss_BA.append(loss_BA.item())

            pre_label = fake_A.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = real_A_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            acc += eval_metrix['mean_class_accuracy']
            miou += eval_metrix['miou']
            class_acc += eval_metrix['class_accuracy']

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                image_save_plot.sample_images_v2(epoch, batches_done, log_dir, real_A=real_A, real_B=real_B,
                                                 fake_A=fake_A,
                                                 fake_B=fake_B, de_transforms_=de_transforms_)

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [G_AB MSE: %f real : %f imag : %f time : %f] [G_BA NLL: %f, miou: %f time : %f] "
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_AB.item(),
                    loss_r.item(),
                    loss_i.item(),
                    time_AB,
                    loss_BA.item(),
                    eval_metrix['miou'],
                    time_BA
                )
            )

        mean_AB = np.mean(eval_loss_AB)
        mean_BA = np.mean(eval_loss_BA)
        mean_AB_r = np.mean(eval_loss_AB_r)
        mean_AB_i = np.mean(eval_loss_AB_i)
        mean_miou = miou / len(dataloader)
        mean_acc = acc / len(dataloader)
        mean_class_acc = class_acc / len(dataloader)
        mean_time_AB = np.mean(times_AB)
        mean_time_BA = np.mean(times_BA)

        print(
            r"Epoch[{:0>3}/{:0>3}]  G_AB_loss:{:.6f} R_loss:{:.6f}  I_loss:{:.6f} time:{:.6f}  G_BA_loss:{:.6f}  Miou :{:.6f}  Train_acc:{:.6f}  time:{:.6f}  ".format(
                epoch, Epoch, mean_AB, mean_AB_r, mean_AB_i,mean_time_AB, mean_BA, mean_miou, mean_acc,mean_time_BA))

        # 绘图
        loss_rec["loss_G_AB"].append(mean_AB),
        loss_rec["loss_G_BA"].append(mean_BA),
        loss_rec["G_BA_Miou"].append(mean_miou),
        loss_rec["G_BA_acc"].append(mean_acc)

        # plt_x = np.arange(1, epoch + 2)
        # image_save_plot.plot_line_v2(plt_x,
        #                              loss_rec["loss_G_AB"], loss_rec["loss_G_BA"],
        #                              loss_rec["G_BA_Miou"], loss_rec["G_BA_acc"],
        #                              out_dir=log_dir)

    print(
        " done ~~~~ {}, best acc: {} in :{} epochs. ".format(
            datetime.datetime.strftime(datetime.datetime.now(), '%m-%d_%H-%M'),
            best_loss, best_epoch))
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
