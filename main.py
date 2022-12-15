from math import inf
import os
import random
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

# 获取图像sift特征描述子
def get_sift_dec(img:np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=50)
    key_points, dec = sift.detectAndCompute(gray, None)
    return dec

# 获取所有的特征向量列表
def get_dec_list(data):
    dec_list = None
    for item in data:
        try:
            dec_list = np.vstack((dec_list, item[0]))
        except:
            dec_list = item[0]
    return dec_list

# 获取图像数据
def get_data(path):
    data = []
    for index, dir in enumerate(os.listdir(path)):
        dirpath = os.path.join(path, dir)
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            img = cv2.imread(filepath)
            data.append([get_sift_dec(img), index])
    return data

# 获取特征值直方图
def get_dec_hist(kmeans, dec_centers, sift_dec):
    hist = [0] * len(dec_centers)
    for pred in kmeans.predict(sift_dec):
        hist[pred] += 1
    return hist

# 获取直方图列表
def get_hist_list(kmeans, dec_centers, data):
    hist_list_with_label = []
    for sift_dec, label in tqdm(data, desc='get_hist_list'):
        hist_list_with_label.append([get_dec_hist(kmeans, dec_centers, sift_dec), label])
    return hist_list_with_label

# 求距离
def L_norm(A, B, way=2):
    ans = 0
    for i in range(len(A)):
        ans += abs(A[i] - B[i]) ** way
    return ans ** (1 / way)

# 获取最近图像
def choose_the_most_similar_image(dec_hist, hist_list_with_label):
    final_dist = inf
    final_label = -1
    for hist, label in hist_list_with_label:
        dist = L_norm(dec_hist, hist)
        if dist < final_dist:
            final_dist = dist
            final_label = label
    return final_label

if __name__ == '__main__':
    dataset_path = './datasets'
    random.seed(0)

    data = get_data(dataset_path)
    train_data, test_data = torch.utils.data.random_split(data, [len(data)-10, 10], generator=torch.Generator().manual_seed(0))
    
    train_dec_list = get_dec_list(train_data)
    kmeans = KMeans(n_clusters=100)
    train_sift_dec_pred = kmeans.fit_predict(train_dec_list)
    sift_dec_centers = kmeans.cluster_centers_
    
    train_dec_hist_list = get_hist_list(kmeans, sift_dec_centers, train_data)
    test_dec_hist_list = get_hist_list(kmeans, sift_dec_centers, test_data)

    predict = []
    label_truth = []
    for item in test_dec_hist_list:
        predict.append(choose_the_most_similar_image(item[0], train_dec_hist_list))
        label_truth.append(item[1])

    print(predict)
    print(label_truth)
    