import pandas as pd

def mtl_composition():
    raw = pd.read_csv("multiTask\cmp-all.csv")
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

if __name__ == '__main__':
    print(mtl_composition().info())