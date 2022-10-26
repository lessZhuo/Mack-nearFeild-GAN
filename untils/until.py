import pandas as pd
import copy
import numpy as np

def save_near_field(path, xy):
    matrix = pd.read_csv(path, usecols=[0], header=None, names=['index'])

    csv_1 = matrix[matrix['index'] == "Array_2D: %i\t%i" % (f, l)].index.tolist()
    # print(label, csv_1, f, l)
    ma = []
    mk = []

    csv_2 = copy.deepcopy(csv_1)
    # print(csv_2)
    for i in range(0, len(csv_2)):
        n = csv_2[i] + 1
        m = csv_2[i] + l + 1
        # print(n, m)
        rows = matrix.iloc[n:m, 0]
        mat = np.array(rows)
        # print(rows, mat)
        for j in range(l):
            # print(l)
            row = mat[j].split()
            # print(row)
            a = list(map(float, row))
            # print(a)
            a_array = np.array(a)
            # print(a_array)
            mat[j] = a_array
            # print(mat[j])

        k = np.stack(mat)
        # print(k)
        ma.append(k)
        mk = np.stack(ma, axis=0)
        # print(mk.shape)

    xx_real = mk[0] * np.cos((mk[4] / 180) * np.pi)
    xx_imag = mk[0] * np.sin((mk[4] / 180) * np.pi)
    yy_real = mk[3] * np.cos((mk[7] / 180) * np.pi)
    yy_imag = mk[3] * np.sin((mk[7] / 180) * np.pi)
    if xy == 0:
        label = xx_real
    elif xy == 1:
        label = xx_imag
    elif xy == 2:
        label = yy_real
    else:
        label = yy_imag
    return label