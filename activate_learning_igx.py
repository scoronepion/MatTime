# 初步思路：先从原始数据集中选取一个质心，再选取距离质心最近和最远的两个样本
import numpy as np
from big_predata import read_element

def calc_cos(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

class node():
    def __init__(self, cos_value, index):
        self.cos_value = cos_value
        self.index = index

if __name__ == '__main__':
    # print(calc_cos(np.array([1,2,3]), np.array(4,5,6)))
    raw = read_element(sort=True).values[:, :]
    print(raw.shape)
    # 质心
    center = raw.mean(axis=0)
    max_index = 0
    max_cos = 0
    for i in range(raw.shape[0]):
        res = calc_cos(raw[i], center)
        if res > max_cos:
            max_cos = res
            max_index = i
    center = raw[max_index]
    # 计算质心与其他数据的夹角
    cos_list = []
    for i in range(raw.shape[0]):
        res = calc_cos(raw[i], center)
        cos_list.append(node(res, i))
    # 从大到小排序
    cos_list.sort(key=lambda x:x.cos_value)
    for item in cos_list:
        print(item.cos_value, item.index)