import numpy as np

data = np.load("D:\dataset\RA_NF_left_up_100_100.npz")
# print(data.size)
print(data["xx_real"].shape)
nf_real = data["xx_real"]
nf_img = data["xx_imag"]

nf = np.concatenate((nf_real[np.newaxis,:,:], nf_img[np.newaxis,:,:]), axis=0)
print(nf.shape)
data = np.load("D:\dataset\A\RA_Mask_left_down_100_100.npy")
print(data.shape)
