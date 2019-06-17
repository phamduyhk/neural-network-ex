#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

##### シグモイド関数, 誤差関数
def sigmoid(x):
    ##### 課題2(a)
    # returnの後にシグモイド関数を返すプログラムを書く
    return

def error_function(x, y):
    ##### 課題2(b)
    # returnの後に誤差関数を返すプログラムを書く
    return
    
    
##### データ数
n = 10000
n_test = 1000

d = 28**2

##### パラメータの初期値
w = np.random.normal(0, 0.3, d+1)

########## 確率的勾配降下法によるパラメータ推定
e = []
e_test = []
error = []
error_test = []

step = 0
num_epoch = 0

eta = 0.01

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    step += 1
    for i in index:
        
        dat = np.genfromtxt("./train/{}.csv".format(i))[1:]
        
        xi = np.append(1, dat[1:])
        yi = dat[0]

        ########## 課題2(c) ここから

        ##### 誤差評価
        e.append(error_function(np.dot(w, xi), yi))
        
        ########## ここまで
    
    ##### エポックごとの訓練誤差
    error.append(sum(e)/n)
    e = []
    
    ##### テスト誤差
    for j in range(0, n_test):        
        dat = np.genfromtxt("./test/{}.csv".format(j))[1:]
        
        xi = np.append(1, dat[1:])
        yi = dat[0]
        
        e_test.append(error_function(np.dot(w, xi), yi))
    
    error_test.append(sum(e_test)/n_test)
    e_test = []
    

########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf")




