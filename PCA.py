# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:04:07 2021

@author: user
"""
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
x_data = iris.data
y_data = iris.target

def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis= 0) #mean()函式對陣列進行求均值運算，其中axis=0表示對各列求均值，返回1* n陣列
    newData = dataMat - meanVal#將樣本資料進行中心化，即對每個維度減去這個維度的資料均值
    return newData, meanVal

def PCA_reduce(dataMat, top):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0) #cov()函式實現的是求出兩個變數的協方差矩陣，得到一個2*2的陣列，其中rowvar=0表示將每一列看作一個變數
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) #eig()函式求解矩陣的特徵值和特徵向量，該函式將返回一個元組，其中第一個元素為特徵值，第二個元素為特徵向量（且每一列為一個特徵向量）
    #print('eigVals',eigVals)
    EV2 = np.cumsum(eigVals)/np.sum(eigVals)
    print('EV2',EV2)
    eigValIndice = np.argsort(eigVals)#將特徵值陣列元素從小到大進行排序，並返回排序後元素的索引
    n_eigValIndice = eigValIndice[-1:-(top + 1): -1]
    n_eigVects = eigVects[:, n_eigValIndice]
    #print(n_eigVects)

# =============================================================================
#     eigValIndice1 = np.argsort(-eigVals)# #將特徵值陣列元素從大到小進行排序，並返回排序後元素的索引
#     #得到最大的n個特徵值的下標
#     print(eigValIndice1)
#     print('l:',eigValIndice1[:top])
#     projection_matrix = eigVects[:,eigValIndice1[:top]]##取排序後的前figureNumber個特徵值所對應的特徵向量組成投影矩陣
#     print(' projection_matrix', projection_matrix)
# =============================================================================

    #得到下標對應的特徵向量
    lowDDataMat = np.dot(newData,n_eigVects)# newData * n_eigVects
    #print(np.dot(newData,n_eigVects))
    # print(lowDDataMat)
    #利用低維度數據來重構數據
    reconMat = np.dot(lowDDataMat , n_eigVects.T) + meanVal# reconstructedData = numpy.dot ( newDataSet, projection_matrix.T ) + meanValue
    #reconMat1 = (lowDDataMat * n_eigVects.T) + meanVal
    return lowDDataMat, reconMat# newDataSet, reconstructedData

# =============================================================================
# def PCATrain(D,R):
#     sn,fn = D.shape
#     meanv = np.mean(D,axis=0)
#     stdv = np.std(D,axis=0)
#     D2 = (D-np.matlib.repmat(meanv,sn,1))/np.matlib.repmat(stdv,sn,1)
#     print('D2',D2)
#     C = np.dot(np.transpose(D2),D2)
#     EValue,Evector = np.linalg.eig(C)
#     print('EValue',EValue)
#     EV2 = np.cumsum(EValue)/np.sum(EValue)
#     print(EV2)
#     num = np.where(EV2>R)[0][0]+1
#     print('num',num)
#     eigValIndice = np.argsort(EValue)# #將特徵值陣列元素從小到大進行排序，並返回排序後元素的索引
#     #得到最大的n個特徵值的下標
#     n_eigValIndice = eigValIndice[-1:-(num + 1): -1]###取排序後的後top個特徵值所對應的特徵向量組成投影矩陣
#     #得到下標對應的特徵向量
#     n_eigVects = Evector[:, n_eigValIndice]
#     #只保留前num維  
#     lowDDataMat = np.dot(D-meanv,n_eigVects)# newData * n_eigVects
#     #print(np.dot(newData,n_eigVects))
#     # print(lowDDataMat)
#     #利用低維度數據來重構數據
#     reconMat = np.dot(lowDDataMat , n_eigVects.T) + meanv# reconstructedData = numpy.dot ( newDataSet, projection_matrix.T ) + meanValue
#     #reconMat1 = (lowDDataMat * n_eigVects.T) + meanVal
#     return lowDDataMat, reconMat# newDataSet, reconstructedData
# lowDDataMat, reconMat =  PCATrain(x_data, 0.8)
# print(lowDDataMat)
# =============================================================================




# =============================================================================
# #直接套PCA套件的結果
# from sklearn.decomposition import PCA
# pca_sk = PCA(n_components=2) 
# newMat = pca_sk.fit_transform(x_data) 
# print(newMat)
#  
# =============================================================================
#將數據降至2維
lowDDataMat, reconMat = PCA_reduce(x_data, 2)
#print(lowDDataMat)
#二維畫圖
x = np.array(lowDDataMat)[:, 0]
y = np.array(lowDDataMat)[:, 1]
plt.scatter(x, y, c = y_data)
plt.show()


#三維
#lowDDataMat, reconMat = PCA_reduce(x_data, 3)

#print(PCATrain(x_data, 0.8))

# =============================================================================
# #矩陣運算example
# a = np.matrix([[3,2,-1],[-2,-2,2],[3,6,-1]])
# eigVals, eigVects = np.linalg.eig(np.mat(a)) #eig()函式求解矩陣的特徵值和特徵向量，該函式將返回一個元組，其中第一個元素為特徵值，
# # 第二個元素為特徵向量（且每一列為一個特徵向量）
# #對特徵值從小到大排列
# #print('eigVals',eigVals)
# print(eigVals)
# EV2 = np.cumsum(eigVals)/np.sum(eigVals)
# print(EV2)
# =============================================================================
