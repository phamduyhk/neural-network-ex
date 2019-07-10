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

# データの取得
# クラス数を定義
m = 4

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train[y_train < m, :, :]

x_test = x_test.astype('float32') / 255.
x_test = x_test[y_test < m, :, :]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

n, d, _ = x_train.shape
n_test, _, _ = x_test.shape

print("n={},d={}".format(n, d))
np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播, 逆伝播


def sigmoid(x):
    tmp = 1/(1+np.exp(-x))
    return tmp, tmp*(1-tmp)


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


def forward(x, param, fncs):
    tmp = np.dot(param, x)
    z = np.append(1, fncs(tmp)[0])
    u = fncs(tmp)[1]
    return z, u


def forward_RNN(x, z, w_in, w_hidden, fncs):
    tmp1 = np.dot(w_in, x)
    # print("w_in {}, x{}, z {}".format(w_in.shape,x.shape,z.shape))
    tmp2 = np.dot(w_hidden, z)
    # print("tmp1 {}, tmp2 {}".format(tmp1.shape,tmp2.shape))
    z = np.append(1,fncs(tmp1+tmp2)[0])
    dz = fncs(tmp1+tmp2)[1]
    return z, dz


def backward(delta, w_hidden, w_out, delta_out, derivative_out):
    print("delta {}, w {}, w_out {}, delta_out {}, der {}".format(
        delta.shape, w_hidden.shape, w_out.shape, delta_out.shape, derivative_out.shape))
    tmp = np.dot(w_hidden.T, delta)+np.dot(w_out.T, delta_out)
    print("tmp {}".format(tmp.shape))
    return np.dot(tmp, derivative_out)


def adam(param, m, v, error, t,
         alpha=0.001, beta1=0.9, beta2=0.999, tol=10**(-8)):
    # returnの後にadamによるパラメータ更新のプログラムを書く
    grad = error
    m = beta1*m+(1.0-beta1)*grad
    v = beta2*v+(1.0-beta2)*grad**2

    m_h = m/(1.0-beta1**t)
    v_h = v/(1.0-beta2**t)

    param -= alpha*(m_h/(np.sqrt(v_h)+tol))
    return param, m, v


# 中間層のユニット数とパラメータの初期値
q = 64

w_in = np.random.normal(0, 0.3, size=(q, d+1))
w_out = np.random.normal(0, 0.3, size=(m, q+1))
w_hidden = np.random.normal(0, 0.3, size=(q, q+1))

# 誤差逆伝播法によるパラメータ推定
num_epoch = 10
eta = 10**(-2)

e = []
e_test = []
error = []
error_test = []


for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    Z = np.zeros(shape=(d+1, q+1))
    U = np.zeros(shape=(d+1, q))
    G = np.zeros(shape=(d+1,m))
    eta_t = eta/(epoch+1)

    for i in index:
        xi = x_train[i, :, :]
        yi = y_train[i, :]

        # 順伝播
        Z[0] = np.append(1,np.zeros(q))
        G[0] = softmax(np.dot(w_out, Z[0]))-yi
        for j in range(d):
            Z[j+1], U[j] = forward_RNN(np.append(1, xi[j, ]),
                                       Z[j], w_in, w_hidden, sigmoid)
            G[j+1] = softmax(np.dot(w_out,Z[j+1]))-yi
             
        z_out = softmax(np.dot(w_out, Z[d]))

        ##### 誤差評価: 誤差をeにappendする
        e.append(CrossEntoropy(z_out, yi))
        # 逆伝播
        delta_out = z_out - yi
        delta = np.zeros(shape=(d+1, q))
        delta[d] = np.zeros(q)
        for j in range(d)[::-1]:
            delta[j] = backward(delta[j+1], w_hidden,
                                w_out[:, 1:], delta_out, U[j])

        # delta = delta[1:,:].T
        # パラメータの更新
        x = np.zeros(shape=(d,d+1))
        for j in range(d):
            x[j,] = np.append(1,xi[j,])
        delta = delta[:d,:]
        print("delta {}".format(delta.shape))
        w_in -= eta_t*np.dot(delta.T,x)
        w_hidden -= eta_t*np.dot(delta.T, Z[1:,: ])
        print("Z {}, G {}".format(Z.shape,G.shape))
        w_out -= eta_t*np.dot(G.T, Z)

    ##### エポックごとの訓練誤差: eの平均をerrorにappendする
    error.append(sum(e)/n)
    print("epoch {} | err {}".format(epoch, sum(e)/n))
    e = []

    # : 誤差をe_testにappendする
    # test error
    for i_test in range(0, n_test):
        xi = x_test[i, :, :]
        yi = y_test[i, :]
        # 順伝播
        Z[0] = np.zeros(q)
        for j in range(d):
            Z[j+1], U[j] = forward_RNN(np.append(1, xi[j, ]),
                                       Z[j], w_in, w_hidden, sigmoid)
        z_out = softmax(np.dot(w_out, np.append(1, Z[d])))
        ##### 誤差評価: 誤差をeにappendする
        e_test.append(CrossEntoropy(z_out, yi))

    ##### エポックごとの訓練誤差: e_testの平均をerror_testにappendする
    error_test.append(sum(e_test)/n_test)
    print("        | test err {}".format(sum(e_test)/n_test))
    e_test = []

# 誤差関数のプロット
plt.plot(error, label="training", lw=3)  # 青線
plt.plot(error_test, label="test", lw=3)  # オレンジ線
plt.grid()
plt.legend(fontsize=16)
plt.savefig("./error.pdf", bbox_inches='tight', transparent=True)

# 確率が高いクラスにデータを分類
prob = []
for j in range(0, n_test):
    xi = x_test[j, :, :]
    yi = y_test[j, :]

    # 順伝播
    Z[0] = np.zeros(q)
    for j in range(d):
        Z[j+1], U[j] = forward(np.append(1, xi[j, ]),
                               Z[j], w_in, w_hidden, sigmoid)

    z_out = softmax(np.dot(w_out, np.append(1, Z[d])))

    prob.append(z_out)

predict = np.argmax(prob, 1)

# confusion matrixと誤分類結果のプロット
ConfMat = np.zeros((m, m))
for i in range(m):
    idx_true = (y_test[:, i] == 1)
    for j in range(m):
        idx_predict = (predict == j)
        ConfMat[i, j] = sum(idx_true*idx_predict)
        if j != i:
            for l in np.where(idx_true*idx_predict == True)[0]:
                plt.clf()
                D = x_test[l, :, :]
                sns.heatmap(D, cbar=False, cmap="Blues", square=True)
                plt.axis("off")
                plt.title('{} to {}'.format(i, j))
                plt.savefig("./misslabeled{}.pdf".format(l),
                            bbox_inches='tight', transparent=True)

plt.clf()
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype=int), linewidths=1,
            annot=True, fmt="1", cbar=False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)
