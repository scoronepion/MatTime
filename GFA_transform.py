import pandas as pd
import re
import random
import pickle
import numpy as np
from decimal import Decimal

def read_data(path=None):
    raw = pd.read_csv(path)

    conpisition_pattern = re.compile(r'([A-Z][a-z]*)([0-9]*\.*[0-9]*)')
    brackets_pattern = re.compile(r'(.*)\((.*)\)([0-9]*\.*[0-9]*)(.*)')

    for item in raw['Alloy']:
        result = brackets_pattern.findall(item)
        if result != []:
            ingredient = Decimal(result[0][2])
            elements = conpisition_pattern.findall(result[0][1])
            tmp = ''
            for elem in elements:
                tmp += elem[0]
                if elem[1] == '':
                    tmp += str(ingredient)
                else:
                    tmp += str(Decimal(elem[1]) * ingredient)
            raw.loc[raw['Alloy'] == item, 'Alloy'] = result[0][0] + tmp + result[0][3]
            # print(raw[raw['Alloy'] == item]['Alloy'])

    print(raw.head())

    features = pd.DataFrame(columns=['Ge', 'Cu', 'Pd', 'Mg', 'Ce', 'Ca', 'Zn', 'Pt', 'Zr', 'P', 
                                    'Y', 'Ni', 'In', 'Sn', 'V', 'Li', 'Sc', 'Be', 'Au', 'Fe', 
                                    'Nb', 'Gd', 'Rh', 'Ir', 'C', 'Ti', 'Co', 'Si', 'Ta', 'Pb', 
                                    'Dy', 'Mo', 'Sm', 'Pr', 'Hf', 'Ga', 'Al', 'Mn', 'Ru', 'Sb', 
                                    'Ho', 'La', 'Tm', 'W', 'Lu', 'U', 'Sr', 'Ag', 'Er', 'B', 'Cr', 
                                    'Yb', 'Nd', 'Ba', 'Tb', 'T'])
    for item in raw['Alloy']:
        details = {}
        result = conpisition_pattern.findall(item)
        for elem in result:
            if elem[1] != '':
                details[elem[0]] = float(elem[1])
            else:
                details[elem[0]] = 1.0
        print(details)
        features = features.append(details, ignore_index=True)

    features = features.fillna(0.0)
    
    phase_onehot = pd.get_dummies(raw['Phase'])
    features = features.join(phase_onehot)
    features['RMG'] = features['BMG'].astype(np.float64)
    features['BMG'] = features['RMG'].astype(np.float64)
    features['CRA'] = features['CRA'].astype(np.float64)

    features['Phase'] = raw['Phase']
    features.loc[features['Phase'] == 'BMG', 'Phase'] = 0
    features.loc[features['Phase'] == 'CRA', 'Phase'] = 1
    features.loc[features['Phase'] == 'RMG', 'Phase'] = 2

    print(features.head())
    print(features.info())

    with open('/home/lab106/zy/MatTime/GFA_trans.pk', 'wb') as f:
        pickle.dump(features, f)

    # raw.to_csv('/home/lab106/zy/MatTime/gfa_trans.csv', index=None)

if __name__ == '__main__':
    read_data(path='/home/lab106/zy/MatTime/gfa.csv')