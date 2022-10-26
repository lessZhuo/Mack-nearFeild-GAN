import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import matplotlib
matplotlib.use('TkAgg')


BASE_DIR = r'E:\Datasets\近场数据\RECT_ARRAY_NF_x0_y0\mat_data'  # 读取MASK的文件位置
path1 = r'E:\Datasets\近场数据\RECT_ARRAY_NF_x0_y0\mat_data'
import os
fileList = os.listdir(path1)
# for i in range(0,5):
#     mat = fileList[i].split("_")
#     f=int(mat[2])
#     last=int(mat[3].split(".")[0])
#
#     file = "RECT_ARRAY_%i_%i.mat" % (f, last)
#     data_dir = os.path.join(BASE_DIR, file)
#     reftracker = scio.loadmat(data_dir)  # 读出来是个字典
#
#     data = reftracker['mask']
#     numpy_data = np.transpose(data)
#
#     x1 = numpy_data[np.newaxis, :]
#     x1 = x1[np.newaxis, :]
#
#     print(x1.shape)
#     #
#     # plt.imshow(numpy_data, cmap="gray")
#     # plt.show()

for i in range(0,5):
    mat = fileList[i].split("_")
    f=int(mat[2])
    last=int(mat[3])
    xy=mat[4]

    file = "RECT_ARRAY_%i_%i_%s_0_0.mat" % (f, last,xy)
    print(file)
    data_dir = os.path.join(BASE_DIR, file)
    reftracker = scio.loadmat(data_dir)  # 读出来是个字典
    print(reftracker)
    data = reftracker['space_%s'%(xy)]
    numpy_data = np.transpose(data)

    x1 = numpy_data[np.newaxis, :]
    x1 = x1[np.newaxis, :]

    print(x1.shape)
    #
    # plt.imshow(numpy_data, cmap="gray")
    # plt.show()