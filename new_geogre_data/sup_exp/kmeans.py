from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd

def read_data():
    raw = pd.read_csv('/home/lab106/zy/new_geogre_data/sup_exp/wyjData.csv')
    raw.drop(columns=['id', 'pic_num', 'formula', 'sTemper', 'sHour', 'sType', 'deltaT'], inplace=True)
    return raw

def kms():
    raw = read_data().values
    instance = KMeans(n_clusters=2)
    y = instance.fit_predict(raw.values)
    print(y)

def tsne(id):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    raw = read_data()
    instance = TSNE()
    instance.fit_transform(raw.values)
    pic = pd.DataFrame(instance.embedding_, index=raw.index)
    for index, row in pic.iterrows():
        plt.plot(row[0], row[1], 'b.')
        plt.text(row[0], row[1], index)
    plt.savefig('/home/lab106/zy/new_geogre_data/sup_exp/test{}.jpg'.format(i))
    plt.show()

if __name__ == '__main__':
    for i in range(10):
        print(i)
        tsne(i)