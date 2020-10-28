import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sko.PSO import PSO

current_model = None

def read_data():
    # 读取csv文件
    raw = pd.read_csv('hn/psodata.csv').dropna().astype('float64').values
    features = raw[:, :-1]
    targets = raw[:, -1]
    return features, targets
    # 设置训练集测试集划分比例（理论上应该需要划分以检验模型效果，但是因数据量过少且重点在于pso，可以不进行划分）
    # x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3)
    # return x_train, x_test, y_train, y_test

def create_and_fit_model(x_train, y_train):
    # 定义模型（输入为15维特征，输出为一维）
    # 注意，这里使用了StandardScaler进行标准化，后续pso搜索的时候，搜出来的x结果未必是真实值，可能是经过标准转化后的值。解决方法：考虑保存当前标准化器，并在搜索结果展示的时候恢复其数据
    model_pz = GridSearchCV(make_pipeline(StandardScaler(),SVR()),param_grid=dict(svr__gamma=[0.1],svr__C=[2000]),cv=3)
    # 训练模型
    model_pz.fit(x_train, y_train)
    global current_model
    current_model = model_pz

def calc_func(x):
    # 打日志，觉得啰嗦可以注释掉
    print('current x = {}'.format(x))
    # 定义搜索范围，请输入要搜索特征的下标（从0开始）
    # 0:THT, 1:THQCr, 2:Dt, 3:QmT, 4:Si, 5:Mn, 6:P, 7:S, 8:Ni, 9:Cu, 10:Mo, 11:NT, 12:CT, 13:Cr, 14:C
    search_range = [11, 12, 13, 14]
    assert len(x) == len(search_range), '输入x长度与搜索范围不相等，请检查'
    # 定义不变数据，这里采用了第一条数据作为数据模板，
    data_template = [[30,0,0,30,0.19,0.42,0.026,0.022,0.01,0.02,0,885,30,0.02,0.22]]
    # 替换数据
    for i in range(len(search_range)):
        data_template[0][search_range[i]] = x[i]
    # 转换成nparray
    pso_x = np.array(data_template)
    # 载入模型
    global current_model
    if current_model == None:
        print('请先调用create_and_fit_model()训练模型')
        return
    else:
        return current_model.predict(pso_x)

def calc_pso():
    # num表示搜索范围，calc_func里的search_range长度必须与它保持一致
    num = 4
    pso = PSO(func=calc_func, dim=num, pop=400, max_iter=400, lb=np.zeros(num), ub=np.ones(num)*100)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

if __name__ == '__main__':
    features, targets = read_data()
    create_and_fit_model(features, targets)
    calc_pso()