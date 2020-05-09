import pandas as pd

def mtl_composition():
    raw = pd.read_csv("multiTask/cmp-all.csv")
    raw['Tx-Tg'] = raw['Tx'] - raw['Tg']
    raw.drop(columns=['sum', 'Tx', 'Tg', 'Tl', 'Phase'], inplace=True)
    # raw.loc[raw['Phase'] == 'BMG', 'Phase'] = 0
    # raw.loc[raw['Phase'] == 'CRA', 'Phase'] = 1
    # raw.loc[raw['Phase'] == 'RMG', 'Phase'] = 2
    # raw['Phase'] = raw['Phase'].astype('int32')
    # tmp = raw.loc[raw['Phase'].isin([0,1])]
    # print(tmp)
    # print(raw.info())
    return raw.dropna().astype('float64')

def mtl_atomic():
    raw = pd.read_csv("multiTask/atomic_full_4_multitask.csv").dropna().astype('float64')
    raw.drop(columns=['Tx', 'Tg'], inplace=True)
    return raw

def composition_atomic_feature():
    com = pd.read_csv("multiTask/cmp-all.csv").dropna()
    ato = pd.read_csv("multiTask/atomic_full_4_multitask.csv").dropna()
    com.drop(columns=['sum', 'Tx', 'Tg', 'Tl', 'Phase', 'Dmax'], inplace=True)
    ato.drop(columns=['Tg', 'Tx'], inplace=True)
    # print(com.astype('float64').info())
    # print(ato.astype('float64').info())
    raw = pd.concat([com.astype('float64'), ato.astype('float64')], axis=1).dropna()
    return raw

if __name__ == '__main__':
    # print(mtl_composition().info())
    # composition_atomic_feature()
    print(mtl_atomic().info())