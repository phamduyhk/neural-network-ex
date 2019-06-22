#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# データの取得
# クラス数
m = 4

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape([60000, 28*28])
x_train = x_train[y_train < m, :]

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape([10000, 28*28])
x_test = x_test[y_test < m, :]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

n, d = x_train.shape
n_test, _ = x_test.shape

np.random.seed(123)

##### シグモイド関数, 誤差関数


def ReLU(x):
    # 課題1(a)
    # ReLUとその微分を, この順番で返すプログラムを書く
    value = 0
    gradient = 0
    if x > 0:
        value = x
        gradient = 1
    return value, gradient


def softmax(x):
    # 課題1(b)
    # ソフトマックス関数を返すプログラムを書く
    return np.exp(x)/np.sum(np.exp(x))


def CrossEntoropy(x, y):
    # 課題1(c)
    # クロスエントロピーを返すプログラムを書く
    return -np.sum(y*np.log(x))


def forward(x, w, fncs):
    # 課題1(d)
    # 順伝播のプログラムを書く
    v = np.dot(w, x)
    z = [1]
    dz = []
    for item in v:
        i, di = fncs(item)
        z.append(i)
        dz.append(di)
    z = np.array(z)
    dz = np.array(dz)
    return z, dz


def backward(w, delta, derivative):
    # 課題1(e)
    # 逆伝播のプログラムを書く
    # print("w: {},delta: {}, der: {}".format(w.shape,delta.shape,derivative.shape))
    return np.dot(w.T,delta)*derivative


# 中間層のユニット数とパラメータの初期値
q = 200
w = np.random.normal(0, 0.3, size=(q, d+1))
v = np.random.normal(0, 0.3, size=(m, q+1))

# 確率的勾配降下法によるパラメータ推定
e = []
e_test = []
error = []
error_test = []

num_epoch = 1

eta = 0.1
loop = 0
for epoch in range(0, num_epoch):
    loop += 1
    index = np.random.permutation(n)

    eta_t = eta/(epoch + 1)
    for i in index:
        xi = np.append(1, x_train[i, :])
        yi = y_train[i, :]
        # 課題2 ここから
        # 順伝播
        # 入力から中間層へ
        z1, u1 = forward(xi, w, ReLU)
        
        # 中間層から出力層へ
   
        print(v)
        z2 = softmax(np.dot(v, z1))

        # 誤差評価
        e.append(CrossEntoropy(z2, yi))

        # 逆伝播
        # delta = softmax(np.dot(v, z1))-yi
        delta = z2 - yi
        derivative = u1

        # print(derivative.shape)
        # print("z1: {} type {},delta: {} type {}".format(z1.shape,type(z1),delta.shape,type(delta)))
        # パラメータの更新
        v = v - eta_t * np.dot(delta.reshape(m,1),z1.reshape(1,q+1))
        print(v)
        V = v[:, 1:]
        w -= eta_t* np.dot(backward(V,delta,derivative).reshape(q,1),xi.reshape(1,d+1))
        # ここまで
        if loop is 3:
            break
    # training error
    error.append(sum(e)/n)
    print("epoch {} | loss {}".format(epoch,sum(e)/n))
    e = []

    # test error
    for j in range(0, n_test):
        xi = np.append(1, x_test[j, :])
        yi = y_train[j, :]

        z1, u1 = forward(xi, w, ReLU)
        z2 = softmax(np.dot(v, z1))

        e_test.append(CrossEntoropy(z2, yi))

    error_test.append(sum(e_test)/n_test)
    e_test = []

# 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)  # 青線
plt.plot(error_test, label="test", lw=3)  # オレンジ線
plt.grid()
plt.legend(fontsize=16)
plt.savefig("./error.pdf")

# 確率が高いクラスにデータを分類
# モデルの出力を評価
prob = []
for j in range(0, n_test):
    xi = np.append(1, x_test[j, :])
    yi = y_train[j, :]

    z1, u1 = forward(xi, w, ReLU)
    z2 = softmax(np.dot(v, z1))

    prob.append(z2)

predict = np.argmax(prob, 1)

# 課題3
# confusion matrixを完成させる
ConfMat = np.zeros((m, m))

plt.clf()
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype=int), linewidths=1,
            annot=True, fmt="1", cbar=False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)

# 誤分類結果のプロット
for i in range(m):
    idx_true = (y_test[:, i] == 1)
    for j in range(m):
        if j != i:
            idx_predict = (predict == j)
            for l in np.where(idx_true*idx_predict == True)[0]:
                plt.clf()
                D = np.reshape(x_test[l, :], (28, 28))
                sns.heatmap(D, cbar=False, cmap="Blues", square=True)
                plt.axis("off")
                plt.title('{} to {}'.format(i, j))
                plt.savefig("./misslabeled{}.pdf".format(l),
                            bbox_inches='tight', transparent=True)

# plt.show()
