import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

pd.set_option('display.max_rows', None)

train1 = pd.read_csv(
    r"E:\Datasets\近场数据\RECT_ARRAY_NF_x0_y0\RECT_ARRAY_70_40.csv",
    usecols=[0], header=None, names=['user_id'])
a = train1[train1['user_id'] == "Array_2D: 70\t40"].index.tolist()
print(a)
ma = []
print(ma)
b = copy.deepcopy(a)
for i in range(0, len(b)):
    n = b[i] + 1
    m = b[i] + 41
    rows = train1.iloc[n:m, 0]
    mat = np.array(rows)
    for j in range(40):
        row = mat[j].split()
        a = list(map(float, row))
        a_array = np.array(a)
        mat[j] = a_array
    k = np.stack(mat)
    ma.append(k)

    mk = np.stack(ma, axis=0)
    # print(mk.shape)
# print(mk[4])
# print(np.cos(np.pi))
xx_real = mk[0] * np.cos((mk[4] / 180) * np.pi)
xx_imag = mk[0] * np.sin((mk[4] / 180) * np.pi)
yy_real = mk[3] * np.cos((mk[7] / 180) * np.pi)
yy_imag = mk[3] * np.sin((mk[7] / 180) * np.pi)
# print(xx_real)
# print(xx_imag)
# print(yy_real)
# print(yy_imag)
print(ma)