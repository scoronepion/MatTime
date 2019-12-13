import pandas as pd
import re
import random
from sklearn.preprocessing import MinMaxScaler

def read_element(noise=False):
    '''返回元素相对原子质量百分比特征与温度'''
    raw = pd.read_csv('/home/lab106/zy/MatTime/ctt_clean.csv')
    quality_table = pd.read_csv('/home/lab106/zy/MatTime/elements.csv')
    features = pd.DataFrame(columns=['Tm', 'Yb', 'Ce', 'Au', 'Ca', 'Tb', 'Dy', 'Ho', 'Sm', 'Mg', 'Si', 'Pr', 'Fe',
                            'Sc', 'P', 'Be', 'Er', 'Li', 'Co', 'Ag', 'Al', 'C', 'Nd', 'Pb', 'Nb', 'Gd',
                            'Ta', 'Lu', 'W', 'Cu', 'Y', 'Mn', 'Sn', 'Cr', 'Ga', 'Pd', 'Hf', 'Ti', 'Zr',
                            'La', 'Ni', 'B', 'Zn', 'In', 'Mo'])

    pattern = re.compile(r'([A-Z][a-z]*)([0-9]*\.*[0-9]*)')
    for item in raw['Alloy']:
        # result 为该行所有元素与其下标的列表：[(元素, 下标), ...]
        result = pattern.findall(item)
        # 求元素质量总和
        sum_quality = 0.0
        for i in range(len(result)):
            element = result[i][0]
            num = result[i][1]
            temp_quality = float(quality_table[quality_table['element'] == element]['quality']) * float(num)
            sum_quality += temp_quality
        # 求各个元素质量百分比
        details = {}
        for i in range(len(result)):
            element = result[i][0]
            num = result[i][1]
            temp_quality = float(quality_table[quality_table['element'] == element]['quality']) * float(num)
            percentage = temp_quality / sum_quality
            details[element] = percentage

        features = features.append(details, ignore_index=True)

    features = features.fillna(0.0)
    features += 1
    if noise:
        features = features.applymap(lambda x: x + abs(random.gauss(0, 0.0001)))

    Tg_scaler = MinMaxScaler()
    Tx_scaler = MinMaxScaler()
    Tl_scaler = MinMaxScaler()

    Tg_trans = pd.DataFrame(Tg_scaler.fit_transform(raw['Tg'].values.reshape(-1, 1)))
    Tx_trans = pd.DataFrame(Tx_scaler.fit_transform(raw['Tx'].values.reshape(-1, 1)))
    Tl_trans = pd.DataFrame(Tl_scaler.fit_transform(raw['Tl'].values.reshape(-1, 1)))

    features['Tg'] = Tg_trans
    features['Tx'] = Tx_trans
    features['Tl'] = Tl_trans

    print(features.head())
    return features.dropna(), (Tg_scaler, Tx_scaler, Tl_scaler)

if __name__ == '__main__':
    read_element(noise=True)