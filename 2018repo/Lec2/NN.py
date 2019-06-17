#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)

##### シグモイド関数, 誤差関数
def ReLU(x):
    ##### 課題1(a)
    # ReLUとその微分を, この順番で返すプログラムを書く
    return 

def softmax(x):
    ##### 課題1(b)
    # ソフトマックス関数を返すプログラムを書く
    return 

def CrossEntoropy(x, y):
    ##### 課題1(c)
    # クロスエントロピーを返すプログラムを書く
    return 

def forward(x, w, fncs):
    ##### 課題1(d)
    # 順伝播のプログラムを書く
    return 

def backward(w, delta, v):
    ##### 課題1(e)
    # 逆伝播のプログラムを書く
    return 

##### データ数
n = 10000
n_test = 1000

d = 28**2
m = 4     #クラス数

label = np.identity(m)

#####中間層のユニット数
q = 200

##### パラメータの初期値 
w = np.random.normal(0, 0.3, size=(q, d+1))
v = np.random.normal(0, 0.3, size=(m, q+1))

########## 確率的勾配降下法によるパラメータ推定
e = []
e_test = []
error = []
error_test = []

num_epoch = 50

eta = 0.1

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    eta_t = eta/(epoch + 1)
    for i in index:
        
        dat = np.genfromtxt("./train/{}.csv".format(i))[1:]
        
        xi = np.append(1, dat[1:])
        yi = label[int(dat[0])]
        
        ########## 課題2 ここから
        ##### 順伝播 
        
        ##### 誤差評価
        e.append(CrossEntoropy(z2, yi))
        
        ##### 逆伝播

        ##### パラメータの更新

        ########## ここまで
    
    ##### training error
    error.append(sum(e)/n)
    e = []
    
    ##### test error
    for j in range(0, n_test):        
        dat = np.genfromtxt("./test/{}.csv".format(j))[1:]
        
        xi = np.append(1, dat[1:])
        yi = label[int(dat[0])]
        
        z1, u1 = forward(xi, w, ReLU)
        z2 = softmax(np.dot(v, z1))
        
        e_test.append(CrossEntoropy(z2, yi))
    
    error_test.append(sum(e_test)/n_test)
    e_test = []


########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf")


##### 学習したモデルの性能評価
prob = []
for j in range(0, n_test):        
    dat = np.genfromtxt("./test/{}.csv".format(j))[1:]
    
    xi = np.append(1, dat[1:])
    yi = label[int(dat[0])]
    
    z1, u1 = forward(xi, w, ReLU)
    z2 = softmax(np.dot(v, z1))
    
    prob.append(z2)

# 確率が高いクラスにデータを分類
predict = np.argmax(prob, 1)

ConfMat = np.zeros((m, m))

##### 課題3
# confusion matrixを完成させる


sns.heatmap(ConfMat.astype(dtype = int), annot = True, fmt="1", cbar =False, cmap="Blues")
plt.savefig("./confusion.pdf")


p0 = predict[0:250]
p1 = predict[250:500]
p2 = predict[500:750]
p3 = predict[750:1000]

for i in range(m):
    p = predict[(250*i):(250*(i+1))]
    for j in np.array(range(0,250))[p!=i]:
        k = 250*i + j
        
        plt.clf()
        dat = np.genfromtxt("./test/{}.csv".format(k))[1:]
        D = np.reshape(dat[1:], (28, 28)).T
        sns.heatmap(D, cbar =False, cmap="Blues", square=True)
        plt.axis("off")
        plt.title('{} to {}'.format(i, p[j]))
        plt.savefig("./misslabeled{}-{}.pdf".format(i, j), bbox_inches='tight')

