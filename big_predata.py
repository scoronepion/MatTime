import pandas as pd
import numpy as np
import re
import random
from sklearn.preprocessing import MinMaxScaler

def get_element_set():
    '''获取所有元素集合'''
    raw = pd.read_csv('dmax.csv')
    elements_set = set()

    pattern = re.compile(r'([A-Z][a-z]*)([0-9]*\.*[0-9]*)')
    for item in raw['Alloy']:
        # result 为该行所有元素与其下标的列表：[(元素, 下标), ...]
        result = pattern.findall(item)
        for i in range(len(result)):
            element = result[i][0]
            elements_set.add(element)
    print(elements_set)

def read_element(noise=False, sort=False, rare_element_scaler=None, nega_sampling=False):
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
    # features += 0.00001
    if noise:
        features = features.applymap(lambda x: x + abs(random.gauss(0, 0.0001)))

    features.dropna(inplace=True)

    features['Dmax'] = raw['Dmax']

    if sort:
        features.sort_values("Dmax", inplace=True)
        features = features.reset_index(drop=True)

    features.dropna(inplace=True)
    # 负采样
    if nega_sampling:
        # 将 dmax 为 0.0 的 1552 条样本负采样为原来的 0.44
        features.drop(features[features['Dmax'] == 0.0].sample(frac=0.56, axis=0).index, inplace=True)
        # 将 dmax 为 0.1 的 3708 条样本负采样为原来的 0.185
        features.drop(features[features['Dmax'] == 0.1].sample(frac=0.815, axis=0).index, inplace=True)

    # # 将 dmax 扩大 10 倍
    # features['Dmax'] *= 10

    # print(features.tail(5))
    # print(features.info())

    print('finish read')
    return features.dropna()

if __name__ == '__main__':
    raw = read_element(sort=True)
    # get_element_set()