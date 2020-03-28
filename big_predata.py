import pandas as pd
import numpy as np
import re
import random
from minepy import MINE
import pickle
from matplotlib import pyplot as plt
import scipy as sc
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
from sklearn.decomposition import PCA

def get_element_set():
    '''获取所有元素集合'''
    raw = pd.read_csv('clean_gfa.csv')
    elements_set = set()

    pattern = re.compile(r'([A-Z][a-z]*)([0-9]*\.*[0-9]*)')
    for item in raw['Alloy']:
        # result 为该行所有元素与其下标的列表：[(元素, 下标), ...]
        result = pattern.findall(item)
        for i in range(len(result)):
            element = result[i][0]
            elements_set.add(element)
    res = list(elements_set)
    res.sort()
    print(res)

def read_element(noise=False, sort=False, rare_element_scaler=None, nega_sampling=False, dmax_scale=None):
    '''直接返回元素含量百分比'''
    print("start reading...")
    raw = pd.read_csv('ctt_clean.csv')

    # # elements for dmax.csv
    # features = pd.DataFrame(columns=['Ba', 'Mn', 'Dy', 'Fe', 'Ca', 'V', 'Al', 'Co', 'Sr', 'Pb', 'T', 'Rh',
    #                                 'In', 'Tm', 'Cr', 'W', 'U', 'Ho', 'Lu', 'Ce', 'Nb', 'Mo', 'C', 'Ni', 
    #                                 'Li', 'Sm', 'Be', 'Pt', 'Ta', 'Er', 'La', 'Y', 'Sb', 'Yb', 'Zr', 'Sc', 
    #                                 'Ir', 'Ag', 'Pr', 'P', 'B', 'Zn', 'Ga', 'Ge', 'Au', 'Sn', 'Cu', 'Nd', 
    #                                 'Tb', 'Si', 'Pd', 'Ru', 'Ti', 'Hf', 'Mg', 'Gd'])

    # elements for ctt_clean.csv
    features = pd.DataFrame(columns=['Zn', 'Ho', 'Cu', 'Be', 'Pd', 'Cr', 'Lu', 'Fe', 'Si', 'Li', 'Sn', 
                                     'Ta', 'Pr', 'In', 'Er', 'C', 'La', 'Pb', 'Ga', 'B', 'Nb', 'Ni', 
                                     'Al', 'Zr', 'Gd', 'Au', 'Ce', 'Tb', 'Dy', 'W', 'Yb', 'Mo', 'Tm', 
                                     'Sc', 'Y', 'Mn', 'P', 'Hf', 'Sm', 'Mg', 'Ti', 'Ag', 'Ca', 'Nd', 'Co'])

    # # 为稀有元素添加权重
    # quality_table = pd.read_csv('/home/lab106/zy/MatTime/elements.csv')
    # if rare_element_scaler:
    #     rare_element = ['Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    #     quality_table.loc[quality_table['element'].isin(rare_element), ['quality']] *= rare_element_scaler

    pattern = re.compile(r'([A-Z][a-z]*)([0-9]*\.*[0-9]*)')
    for item in raw['Alloy']:
        # result 为该行所有元素与其下标的列表：[(元素, 下标), ...]
        result = pattern.findall(item)
        # print(result)
        # 求元素含量总和
        sum_quality = 0.0
        for i in range(len(result)):
            element = result[i][0]
            num = result[i][1]
            if num == '':
                num = 1.0
            sum_quality += float(num)
        # 求各个元素质量百分比
        details = {}
        for i in range(len(result)):
            element = result[i][0]
            num = result[i][1]
            if num == '':
                num = 1
            percentage = float(num) / sum_quality
            details[element] = percentage

        features = features.append(details, ignore_index=True)
    
    features = features.fillna(0.0)
    if noise:
        features = features.applymap(lambda x: x + abs(random.gauss(0, 0.0001)))

    features.dropna(inplace=True)

    features['Dmax'] = raw['Dmax']

    if sort:
        features.sort_values("Dmax", inplace=True)
        features = features.reset_index(drop=True)
    # print(features.iloc[590:,:].info())
    features.dropna(inplace=True)
    # 负采样
    if nega_sampling:
        # 将 dmax 为 0.0 的 1552 条样本负采样为原来的 0.44
        features.drop(features[features['Dmax'] == 0.0].sample(frac=0.56, axis=0).index, inplace=True)
        # 将 dmax 为 0.1 的 3708 条样本负采样为原来的 0.185
        features.drop(features[features['Dmax'] == 0.1].sample(frac=0.815, axis=0).index, inplace=True)

    # 将 dmax 扩大 dmax_scale 倍
    if dmax_scale:
        features['Dmax'] *= dmax_scale

    # print(features.tail(5))
    # print(features.info())

    print('finish read')
    return features.dropna()

def read_gfa_element(nega_sampling=False):
    '''直接返回元素含量百分比'''
    print("start reading...")
    raw = pd.read_csv('clean_gfa.csv')

    features = pd.DataFrame(columns=['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'C', 'Ca', 'Ce', \
                                    'Co', 'Cr', 'Cu', 'Dy', 'Er', 'Fe', 'Ga', 'Gd', 'Ge', \
                                    'Hf', 'Ho', 'In', 'Ir', 'La', 'Li', 'Lu', 'Mg', 'Mn', \
                                    'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pd', 'Pr', 'Pt', \
                                    'Rh', 'Ru', 'Sb', 'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'T', \
                                    'Ta', 'Tb', 'Ti', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr'])

    pattern = re.compile(r'([A-Z][a-z]*)([0-9]*\.*[0-9]*)')
    for item in raw['Alloy']:
        # result 为该行所有元素与其下标的列表：[(元素, 下标), ...]
        result = pattern.findall(item)
        # print(result)
        # 求元素含量总和
        sum_quality = 0.0
        for i in range(len(result)):
            element = result[i][0]
            num = result[i][1]
            if num == '':
                num = 1.0
            sum_quality += float(num)
        # 求各个元素质量百分比
        details = {}
        for i in range(len(result)):
            element = result[i][0]
            num = result[i][1]
            if num == '':
                num = 1
            # percentage = float(num) / sum_quality
            details[element] = float(num)

        features = features.append(details, ignore_index=True)
    
    features = features.fillna(0.0)

    features['Phase'] = raw['Phase']
    # res = features.join(pd.get_dummies(raw[["Phase"]]))
    features.loc[features['Phase'] == 'BMG', 'Phase'] = 0
    features.loc[features['Phase'] == 'CRA', 'Phase'] = 1
    features.loc[features['Phase'] == 'RMG', 'Phase'] = 2

    # 负采样
    if nega_sampling:
        # 将 dmax 为 0.0 的 1552 条样本负采样为原来的 0.44
        # features.drop(features[features['Phase'] == 1].sample(frac=0.56, axis=0).index, inplace=True)
        # 将 dmax 为 0.1 的 3708 条样本负采样为原来的 0.32
        features.drop(features[features['Phase'] == 2].sample(frac=0.5, axis=0).index, inplace=True)

    print('finish read')
    return features.dropna()

def read_over_element():
    print("start reading...")
    raw = pd.read_csv('trans_dmax.csv')
    print('finish read')
    return raw.dropna()

def read_cmp():
    print('start reading...')
    raw = pd.read_csv('Full_Dmax_Cmp.csv')
    # print(raw.astype('float64').info())
    return raw.astype('float64')

def read_pro_features():
    '''读取计算后特征'''
    print('Start reading...')
    raw = pd.read_csv('full-Dmax.csv')
    raw.drop(raw[raw['Dmax'] == 0].index, inplace=True)
    raw.drop(raw[raw['Dmax'] == 0.1].index, inplace=True)
    del raw['Alloy']
    del raw['Tg']
    del raw['Tx']
    del raw['Tl']
    # raw = raw.iloc[:, [10, 28, 30, 17, 18, 2, 0, 27, 16, 32]]
    print(raw.shape)

    # print(raw[['VEC1', 'sVEC', 'Hfd', 'Tb2', 'Gp1', 'Wd', 'Dmax']].info())
    # return raw[['VEC1', 'sVEC', 'Hfd', 'Tb2', 'Gp1', 'Wd', 'Dmax']]
    return raw

def calc_pac(num):
    raw = read_pro_features()
    pca = PCA(n_components=num)
    # print(raw.drop('Dmax', axis=1).info())
    new_feature = pca.fit_transform(raw.drop('Dmax', axis=1).values)
    dmax = np.expand_dims(raw['Dmax'].values, axis=-1)
    new_raw = np.hstack((new_feature, dmax))
    print(new_raw)
    print(new_raw.shape)
    return new_raw

def calc_mic():
    raw = read_atomic_features()
    length = raw.shape[1]
    res = np.zeros(shape=(length, length))
    i = 0
    for item1 in raw.columns:
        print(i)
        j = 0
        for item2 in raw.columns:
            m = MINE()
            m.compute_score(raw[item1], raw[item2])
            res[i][j] = m.mic()
            j += 1
        i += 1
    print(sorted(enumerate(res[-1]), key=lambda x: x[1]))
    print(res)
    with open('dmax_atomic_features_94_no_negaSampling_mic.b', 'wb') as f:
        pickle.dump(res, f)
    # print(m.mic())

def calc_pear():
    raw = read_atomic_txtg()
    length = raw.shape[1]
    res = np.zeros(shape=(length, length))
    i = 0
    for item1 in raw.columns:
        print(i)
        j = 0
        for item2 in raw.columns:
            m = MINE()
            res[i][j], _ = sc.stats.pearsonr(raw[item1].values, raw[item2].values)
            j += 1
        i += 1
    print(sorted(enumerate(res[-1]), key=lambda x: x[1]))
    # with open('atomic_features_txtg_pear.b', 'wb') as f:
    #     pickle.dump(res, f)
    # print(m.mic())

def pic():
    with open('pro_features_pear.b', 'rb') as f:
        raw = pickle.load(f)
        print(raw[-1])
        plt.imshow(raw)
        plt.savefig('./pics/pear_profeatures.png', dpi=500)
        plt.show()

def read_atomic_features():
    print("start reading...")
    raw = pd.read_csv('Full-Dataset-Dmax.csv')
    print('finish read')
    # 变换dmax
    raw.loc[raw['Dmax'] == 0, 'Dmax'] = -2
    raw.loc[raw['Dmax'] == 0.1, 'Dmax'] = 0.001

    raw = raw[~raw['Dmax'].isin([72])]
    
    # 将RMG 的 3708 条样本负采样为原来的 0.45
    # raw.drop(raw[raw['Phase Formation'] == 'RMG'].sample(frac=0.55, axis=0).index, inplace=True)

    raw.drop(['Phase Formation', 'Alloy Formula'], axis=1, inplace=True)

    # print (raw.shape)
    # collist = [1, 5, 7, 8, 10, 12, 13, 14, 15, 16, 17, 19, 27, 29, 31, 33, 35, 41, 42, 43, 45, 47, 49, 55, 58, 59, 61, 63, 65, 71, 76, 78, 79, 80, 81, 83, 86, 87, 89, 91]

    # # cross_attention_atomic_081.bin保存的是[89, 88, 85, 63, 33, 50, 59, 15, 87, 66, 92, 61, 49, 75, 34, 83, 52, 43, 32, 93, 69, 90, 84, 51, 94]
    # /home/lab106/zy/MatTime/models/cross_attention_atomic_select_feature_13_07502.bin 存的是[12,13,32,33,54,66,70,74,75,76,84,85,86,94]
    # return raw.iloc[:, [12,13,32,33,54,66,70,74,75,76,84,85,86,94]].dropna()

    # 投票法：[11,70,75,64,52,4,92,37,68,69,9,32,36,48,85,94]
    # [4,92,37,68,69,9,32,36,48,85,94]: 0.7101
    # return raw.iloc[:, [4,92,37,68,69,9,32,36,48,85,94]].dropna()
    return raw.dropna()

def read_atomic_features_30():
    print("start reading...")
    raw = pd.read_csv('Full-Dataset-Dmax-30.csv')
    print('finish read')
    raw.drop(['Tg', 'Tx', 'Tl', 'Alloy', 'Class', 'N'], axis=1, inplace=True)

    raw.drop(raw[raw['Dmax'] == 0.1].sample(frac=0.55, axis=0).index, inplace=True)
    return raw

def read_atomic_txtg():
    print("start reading...")
    raw = pd.read_csv('Full_Dataset-Dmax-TTT.csv')
    print('finish read')
    raw.dropna(inplace=True)
    Tx_Tg = raw['Tx'] - raw['Tg']
    raw['Tx_Tg'] = Tx_Tg
    raw.drop(['Phase Formation', 'AAAAAlloy Formula', 'Tg', 'Tx', 'Tl', 'Dmax'], axis=1, inplace=True)
    # return raw.iloc[:, [89,88,85,63,33,50,59,15,87,66,92,61,49,75,34,83,52,43,32,93,69,90,84,51,98]].dropna()
    # 全数据集：0.7112
    # 互信息[39,64,72,81,34,21,78,19,65,80,9,1,69,91,13,52,31,50,90,73,23,37,35,71,67,94]:0.8015
    # return raw.iloc[:, [90,73,23,37,35,71,67,94]].dropna()
    return raw.dropna()


if __name__ == '__main__':
    # raw = read_element(sort=True)
    # raw.to_csv('trans_dmax.csv', index=False)
    # print(raw.info())
    # get_element_set()
    # read_pro_features()
    # calc_pear()
    # pic()
    # calc_pac()
    # raw = read_pro_features()
    # raw = read_cmp()
    # print(raw.tail())
    # raw = read_atomic_features()
    # print(raw.info())
    # raw.to_csv('element_gfa_3_nega_samp.csv', index=False)
    # calc_mic()
    calc_mic()