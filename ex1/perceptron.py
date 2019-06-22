#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from loadnpy import load


##### データの取得
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape([60000, 28*28])
x_train = x_train[y_train <= 1,:]

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape([10000, 28*28])
x_test = x_test[y_test <= 1,:]

y_train = y_train[y_train <= 1]
y_test = y_test[y_test <= 1]

n, d = x_train.shape
n_test, _ = x_test.shape

np.random.seed(123)

##### シグモイド関数, 誤差関数
def sigmoid(x):
    # returnの後にシグモイド関数を返すプログラムを書く
    return 1/(1+np.exp(-x))

def error_function(x, y):
    fx = sigmoid(x)
    return (-y*np.log(fx)-(1-y)*np.log(1-fx))

##### パラメータの初期値
# w = np.random.normal(0, 0.3, d+1)
w = load('weight.npy')

########## 確率的勾配降下法によるパラメータ推定
e = []
e_test = []
error = []
error_test = []

step = 0
num_epoch = 10

eta = 0.01

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    step += 1
    for i in index:
        xi = np.append(1, x_train[i, :]) # add bias
        yi = np.array(y_train[i], float)

        ########## 課題2(c) ここから
        ##### モデルの出力
        y_prediction = sigmoid(np.dot(w,xi))
        # print("モデルの出力 {}".format(epoch,y_prediction))

        ##### 誤差評価
        e.append(error_function(np.dot(w, xi), yi))
        
        ##### パラメータの更新
        ########## 確率的勾配降下法の更新式を書く
        w = w - eta*(sigmoid(np.dot(w,xi))-yi)*xi
        # w = w-eta*(y_prediction-yi)*xi

        ########## ここまで
    
    ##### エポックごとの訓練誤差
    error.append(sum(e)/n)
    print("epoch {} | err {}".format(epoch,sum(e)/n))
    e = []



    ##### テスト誤差
    for j in range(0, n_test):        
        
        xi = np.append(1, x_test[j, :])
        yi = np.array(y_test[j], float)
        
        e_test.append(error_function(np.dot(w, xi), yi))
    error_test.append(sum(e_test)/n_test)
    print("        | test err {}".format(sum(e_test)/n_test))
    e_test = []

########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", transparent=True)
plt.show()

#save weight
np.save('weight.npy',w)





