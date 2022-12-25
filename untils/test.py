import numpy as np
import pandas as pd
import scipy.io as scio


if __name__ == '__main__':

    mask_data = scio.loadmat(r"E:\下载目录\近场数据\RECT_ARRAY_mat_data\RECT_ARRAY_70_40_xy_0_0.mat")
    print(mask_data)