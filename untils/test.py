import numpy as np
import pandas as pd


if __name__ == '__main__':

    df = pd.read_excel(open(r'C:\Users\Administrator\Documents\WeChat Files\wxid_em9fjmpg9twr11\FileStorage\File\2022-12\卡游奥特曼橡皮公仔.xlsx','rb'),sheet_name='sku表')

    print(df)