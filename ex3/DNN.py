#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import time

##### データの取得
#クラス数を定義
m = 4

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape([60000, 28*28])
x_train = x_train[y_train < m,:]

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape([10000, 28*28])
x_test = x_test[y_test < m,:]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

n, d = x_train.shape
n_test, _ = x_test.shape

np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播, 逆伝播
def sigmoid(x):
    value = 1/(1+np.exp(-x))
    # gradient = np.exp(-x) / ((1+np.exp(-x))**2)
    gradient = value*(1-value)
    return value,gradient

def ReLU(x):
    value = 0
    gradient = 0
    if x > 0:
        value = x
        gradient = 1
    return value, gradient

def tanh(x):
    value = np.tanh(x)
    gradient = 1 - value**2
    return value, gradient

def softmax(x):
    e = np.exp(x)
    return e/np.sum(e)


def CrossEntoropy(x, y):
    return -np.sum(y*np.log(x))


def forward(x, w, fncs):
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
    return np.dot(w.T, delta)*derivative

##### 中間層のユニット数とパラメータの初期値
"""" model and parameter
z1 = w1x        (q1,)   #bias be added
z2 = w2z1       (q2+1,) #bias be added
z3 = y = w3z2   (m, )

"""
q1 = 100
q2 = 100
w1 = np.random.normal(0, 0.3, size=(q1, d+1))
w2 = np.random.normal(0, 0.3, size=(q2, q1+1))
w3 = np.random.normal(0, 0.3, size=(m, q2+1))

########## 確率的勾配降下法によるパラメータ推定
# num_epoch = 50
num_epoch = 10

eta = 10**(-2)

e = []
e_test = []
error = []
error_test = []
start = time.time()
for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    eta_t = eta/(epoch+1) 
    for i in index:
        xi = np.append(1, x_train[i, :])
        yi = y_train[i, :]

        ##### 順伝播
        z1, u1 = forward(xi, w1, ReLU)
        z2, u2 = forward(z1, w2, tanh)
        z3 = softmax(np.dot(w3, z2))
        ##### 誤差評価
        e.append(CrossEntoropy(z3, yi))
        ##### 逆伝播
        delta3 = z3 - yi
        v3 = w3[:, 1:]
        v2 = w2[:, 1:]
        delta2 = backward(v3,delta3,u2)
        delta = backward(v2,delta2,u1)
        # パラメータの更新
        w3 -= eta_t*np.outer(delta3, z2)
        w2 -= eta_t*np.outer(delta2,z1)
        w1 -= eta_t*np.outer(delta,xi)
        # w -= eta_t*np.outer(delta2, xi)

    ##### エポックごとの訓練誤差: eの平均をerrorにappendする
    error.append(sum(e)/n) 
    print("epoch {} | err {}".format(epoch, sum(e)/n))
    e = []
    
    ##### test error
    for j in range(0, n_test):
        xi = np.append(1, x_test[j, :])
        yi = y_test[j, :]
        ##### 順伝播
        z1, u1 = forward(xi, w1, ReLU)
        z2, u2 = forward(z1, w2, ReLU)
        z3 = softmax(np.dot(w3, z2))
        ##### テスト誤差: 誤差をe_testにappendする
        e_test.append(CrossEntoropy(z3, yi))
    ##### エポックごとの訓練誤差: e_testの平均をerror_testにappendする
    error_test.append(sum(e_test)/n_test)
    print("        | test err {}".format(sum(e_test)/n_test))
    e_test = []
end = time.time()
print("実行時間： {}".format(end-start))

########## 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", bbox_inches='tight', transparent=True)

########## 確率が高いクラスにデータを分類
prob = []
for j in range(0, n_test):    
    xi = np.append(1, x_test[j, :])
    yi = y_test[j, :]
    
    # テストデータに対する順伝播: 順伝播の結果をprobにappendする
    z1, u1 = forward(xi, w1, ReLU)
    z2, u2 = forward(z1, w2, ReLU)
    z3 = softmax(np.dot(w3, z2))
    prob.append(z3)
   
predict = np.argmax(prob, 1)

# ##### confusion matrixと誤分類結果のプロット
# ConfMat = np.zeros((m, m))
# for i in range(m):
#     idx_true = (y_test[:, i]==1)
#     for j in range(m):
#         idx_predict = (predict==j)
#         ConfMat[i, j] = sum(idx_true*idx_predict)
#         if j != i:
#             for l in np.where(idx_true*idx_predict == True)[0]:
#                 plt.clf()
#                 D = np.reshape(x_test[l, :], (28, 28))
#                 sns.heatmap(D, cbar =False, cmap="Blues", square=True)
#                 plt.axis("off")
#                 plt.title('{} to {}'.format(i, j))
#                 plt.savefig("./misslabeled{}.pdf".format(l), bbox_inches='tight', transparent=True)

# plt.clf()
# fig, ax = plt.subplots(figsize=(5,5),tight_layout=True)
# fig.show()
# sns.heatmap(ConfMat.astype(dtype = int), linewidths=1, annot = True, fmt="1", cbar =False, cmap="Blues")
# ax.set_xlabel(xlabel="Predict", fontsize=18)
# ax.set_ylabel(ylabel="True", fontsize=18)
# plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)
