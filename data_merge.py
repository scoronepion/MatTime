import pandas as pd

if __name__ == '__main__':
    gfa = pd.read_csv('/home/lab106/zy/MatTime/origin_gfa.csv')
    dmax = pd.read_csv('/home/lab106/zy/MatTime/dmax.csv')
    merge = pd.merge(gfa, dmax, on='Alloy')
    print(merge.head())
    print(merge.info())