#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

##### データの取得
#クラス数を定義
m = 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train[y_train < m, :, :]

x_test = x_test.astype('float32') / 255.
x_test = x_test[y_test < m, :, :]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

n, d , _ = x_train.shape
n_test, _, _ = x_test.shape

np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播, 逆伝播

##### 中間層のユニット数とパラメータの初期値

########## 誤差逆伝播法によるパラメータ推定
num_epoch = 10

e = []
e_test = []
error = []
error_test = []

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    for i in index:
        xi = x_train[i, :, :]
        yi = y_train[i, :]
        
        ##### 順伝播
        
        ##### 誤差評価: 誤差をeにappendする
        
        ##### 逆伝播
        
        ##### パラメータの更新

    ##### エポックごとの訓練誤差: eの平均をerrorにappendする

    e = []
    
    #####: 誤差をe_testにappendする
    
    ##### エポックごとの訓練誤差: e_testの平均をerror_testにappendする
    
    e_test = []

########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", bbox_inches='tight', transparent=True)

########## 確率が高いクラスにデータを分類
prob = []
for j in range(0, n_test):    
    xi = x_test[j, :, :]
    yi = y_test[j, :]
    
    ##### 順伝播
    Z[0] = np.zeros(N)
    for j in range(d):
        Z[j+1], U[j] = forward(np.append(1, xi[j,]), Z[j], w_in, w_hidden, sigmoid)
        
    z_out = softmax(np.dot(w_out, np.append(1, Z[d])))
    
    prob.append(z_out)

predict = np.argmax(prob, 1)

##### confusion matrixと誤分類結果のプロット
ConfMat = np.zeros((m, m))
for i in range(m):
    idx_true = (y_test[:, i]==1)
    for j in range(m):
        idx_predict = (predict==j)
        ConfMat[i, j] = sum(idx_true*idx_predict)
        if j != i:
            for l in np.where(idx_true*idx_predict == True)[0]:
                plt.clf()
                D = x_test[l, :, :]
                sns.heatmap(D, cbar =False, cmap="Blues", square=True)
                plt.axis("off")
                plt.title('{} to {}'.format(i, j))
                plt.savefig("./misslabeled{}.pdf".format(l), bbox_inches='tight', transparent=True)

plt.clf()
fig, ax = plt.subplots(figsize=(5,5),tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype = int), linewidths=1, annot = True, fmt="1", cbar =False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)
