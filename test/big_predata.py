import pandas as pd
import numpy as np
import re
import random
#from minepy import MINE
import pickle
from matplotlib import pyplot as plt
import scipy as sc
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def read_atomic_features():
    print("start reading...")
    raw = pd.read_csv('Full-Dataset-Dmax.csv')
    print('finish read')
    # 变换dmax
    raw.loc[raw['Dmax'] == 0, 'Dmax'] = -2
    raw.loc[raw['Dmax'] == 0.1, 'Dmax'] = 0.001
    
    # 将RMG 的 3708 条样本负采样为原来的 0.45
    #raw.drop(raw[raw['Phase Formation'] == 'RMG'].sample(frac=0.55, axis=0).index, inplace=True)

    raw.drop(['Phase Formation', 'Alloy Formula'], axis=1, inplace=True)

    # print (raw.shape)
    # collist = [1, 5, 7, 8, 10, 12, 13, 14, 15, 16, 17, 19, 27, 29, 31, 33, 35, 41, 42, 43, 45, 47, 49, 55, 58, 59, 61, 63, 65, 71, 76, 78, 79, 80, 81, 83, 86, 87, 89, 91]

    return raw.iloc[:, [89, 88, 85, 63, 33, 50, 59, 15, 87, 66, 92, 61, 49, 75, 34, 83, 52, 43, 32, 93, 69, 90, 84, 51, 94]].dropna()
    #return raw.dropna()

if __name__ == '__main__':
    raw = read_atomic_features()
    print(raw.info())
    print(raw)