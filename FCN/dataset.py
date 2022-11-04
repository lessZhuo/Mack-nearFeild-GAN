import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob


class LoadDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", combine=False, direction="x", part="real"):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为csv
        """
        self.transform = transforms.Compose(transforms_)
        self.combine = combine
        self.direction = direction
        self.part = part
        # 获取该文件夹所有符合模式匹配格式的文件，变成list返回
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*.*"))

    def __getitem__(self, index):
        # 1.读取mask的数据
        mask = np.load(self.files_A[index % len(self.files_A)])
        mask = mask[np.newaxis, :, :]
        mask = mask.astype(np.float)
        mask = self.transform(mask)

        # 2.读取近场数据
        near_field = np.load(self.files_B[index % len(self.files_B)])

        # 2.1判读读取xx方向还是yy方向数据
        if self.direction == "x":
            nf_real = near_field["xx_real"]
            nf_imag = near_field["xx_imag"]
        else:
            nf_real = near_field["yy_real"]
            nf_imag = near_field["yy_imag"]

        # 2.2进行类型转换和维度扩充
        nf_real = nf_real.astype(np.float)
        nf_imag = nf_imag.astype(np.float)
        nf_real = nf_real[np.newaxis, :, :]
        nf_imag = nf_imag[np.newaxis, :, :]

        # 2.3根据是否合并或者读取实部虚部进行返回数据
        if self.combine:
            nf = np.concatenate((nf_real, nf_imag), axis=0)
            nf = self.transform(nf)
            return {"A": mask, "B": nf}
        else:
            if self.part == "real":
                return {"A": mask, "B": self.transform(nf_real)}
            else:
                return {"A": mask, "B": self.transform(nf_imag)}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
