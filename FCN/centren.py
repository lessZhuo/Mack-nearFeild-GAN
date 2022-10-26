import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import copy

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import torchvision
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
crop_obj = torchvision.transforms.CenterCrop((300, 300))

BASE_DIR = r'E:\Datasets\近场数据\RECT_ARRAY_MASK'  # 读取MASK的文件位置
BASE_DIR1 = r"E:\Datasets\近场数据\RECT_ARRAY_NF_x0_y0"
path1 = r'E:\Datasets\近场数据\RECT_ARRAY_MASK'

fileList = os.listdir(path1)
for i in range(0, 696):
    mat = fileList[i].split("_")
    f = int(mat[2])
    last = int(mat[3].split(".")[0])
    file = "RECT_ARRAY_%i_%i.mat" % (f, last)
    file1 = "RECT_ARRAY_%i_%i.csv" % (f, last)
    root = 'E:/Datasets/RECT_'
    imgname = "RECT_ARRAY_%i_%i" % (f, last)
    print(file)
    data_dir = os.path.join(BASE_DIR, file)
    data_dir1 = os.path.join(BASE_DIR1, file1)
    train1 = pd.read_csv(
        data_dir1,
        usecols=[0], header=None, names=['user_id'])
    reftracker = scio.loadmat(data_dir)  # 读出来是个字典
    data = reftracker['mask']
    numpy_data = np.transpose(data)

    a = train1[train1['user_id'] == "Array_2D: %i\t%i" % (f, last)].index.tolist()
    ma = []
    b = copy.deepcopy(a)
    for i in range(0, len(b)):
        n = b[i] + 1
        m = b[i] + last + 1
        rows = train1.iloc[n:m, 0]
        mat = np.array(rows)
        for i in range(last):
            row = mat[i].split()
            a = list(map(float, row))
            a_array = np.array(a)
            mat[i] = a_array
        k = np.stack(mat)
        ma.append(k)

        mk = np.stack(ma, axis=0)

    xx_real = mk[0] * np.cos((mk[4] / 180) * np.pi)  # ??????????????????????????
    xx_imag = mk[0] * np.sin((mk[4] / 180) * np.pi)
    yy_real = mk[3] * np.cos((mk[7] / 180) * np.pi)
    yy_imag = mk[3] * np.sin((mk[7] / 180) * np.pi)
    # print(xx_real.shape)
    # print(xx_imag.shape)
    # print(yy_real.shape)
    # print(numpy_data.shape)
    if (300 // f) % 2 == 0:  # ????????????????????????
        v = 300 // f + 1
    else:
        v = 300 // f + 2
    if (300 // last) % 2 == 0:
        h = 300 // last + 1
    else:
        h = 300 // last + 2
    for i in range(h):
        if i == 0:
            num = numpy_data
            xx_num = xx_real
            xx_inum = xx_imag
            yy_num = yy_real
            yy_inum = yy_imag

        else:
            num = np.hstack((num, numpy_data))
            xx_num = np.hstack((xx_num, xx_real))
            xx_inum = np.hstack((xx_inum, xx_imag))
            yy_num = np.hstack((yy_num, yy_real))
            yy_inum = np.hstack((yy_inum, yy_imag))

        if i == (h - 1):
            for j in range(v):
                if j == 0:
                    num_v = num
                    xx_num_v = xx_num
                    yy_num_v = yy_num
                    xx_inum_v = xx_inum
                    yy_inum_v = yy_inum
                else:
                    num_v = np.vstack((num_v, num))
                    xx_num_v = np.vstack((xx_num_v, xx_num))
                    yy_num_v = np.vstack((yy_num_v, yy_num))
                    xx_inum_v = np.vstack((xx_inum_v, xx_inum))
                    yy_inum_v = np.vstack((yy_inum_v, yy_inum))
    # num_v=num_v[:,:,None]

    xx_i_r = xx_inum_v.shape[0]  # ??????????????????
    xx_i_v = xx_inum_v.shape[1]

    img_r1 = int(xx_i_r / 2 - 150)
    img_r2 = int(xx_i_r / 2 + 150)
    img_v1 = int(xx_i_v / 2 - 150)
    img_v2 = int(xx_i_v / 2 + 150)

    xx_i = xx_inum_v[img_r1:img_r2, img_v1:img_v2]

    img_pil = Image.fromarray(num_v)
    img_pil_xx_r = Image.fromarray(xx_num_v)
    # img_pil_xx_i = Image.fromarray(xx_inum_v)
    # img_pil_yy_r = Image.fromarray(yy_num_v)
    # img_pil_yy_i = Image.fromarray(yy_inum_v)
    # img_pil_xx_r = xx_num_v.to.tensor
    # img_pil_xx_i = Image.fromarray(xx_inum_v)
    # img_pil_yy_r = Image.fromarray(yy_num_v)
    # img_pil_yy_i = Image.fromarray(yy_inum_v)
    img_pil_xx_r=crop_obj(img_pil_xx_r)
    # img_pil_xx_i=crop_obj(img_pil_xx_i)
    # img_pil_yy_r=crop_obj(img_pil_yy_r)
    # img_pil_yy_i=crop_obj(img_pil_yy_i)

    img_pil = crop_obj(img_pil)
    xx_r=np.array(img_pil_xx_r)
    # xx_i=np.array(img_pil_xx_i)
    # yy_r=np.array(img_pil_yy_r)
    # yy_i=np.array(img_pil_yy_i)

    # num_v = crop_obj(num_v)
    # print(img_pil.shape)
    plt.axis('off')
    plt.figure(figsize=(3, 3), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 输出图像#边框设置
    plt.imshow(xx_r)
    # img_pil.save(os.path.join(root, imgname + ".jpg"))
    plt.savefig(os.path.join(root, imgname +"xxr" + ".jpg"), pad_inches=0.0)
    plt.show()
    # print(xx_r)
    print(xx_i)
    # print(yy_r)
    # print(yy_i)

    # root = 'E:\Datasets\result'
    # imgname = "RECT_ARRAY_%i_%i" % (f, last)
    # img_pil.save(os.path.join(root, imgname + ".jpg"))
