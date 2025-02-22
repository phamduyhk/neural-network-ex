
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2回演習問題
"""
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
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
    value = x*(x > 0)
    gradient = 1*(x > 0)
    return value, gradient


def sigmoid(x):
    tmp = 1/(1+np.exp(-x))
    return tmp, tmp*(1-tmp)


def softmax(x):
    # 課題1(b)
    # ソフトマックス関数を返すプログラムを書く
    e = np.exp(x)
    return e/np.sum(e)




def CrossEntoropy(x, y):
    # 課題1(c)
    # クロスエントロピーを返すプログラムを書く
    return -np.sum(y*np.log(x))


def forward(x, param, fncs):
    tmp = np.dot(param, x)
    z = np.append(1, fncs(tmp)[0])
    u = fncs(tmp)[1]
    return z, u


def backward(param, delta, derivative):
    return np.dot(param.T, delta)*derivative


def adam(param, m, v, grad, t,
         alpha=0.001, beta1=0.9, beta2=0.999, tol=10**(-8)):
    # returnの後にadamによるパラメータ更新のプログラムを書く
    m = beta1*m+(1.0-beta1)*grad
    v = beta2*v+(1.0-beta2)*grad*grad
    m_h = m/(1.0-beta1**t)
    v_h = v/(1.0-beta2**t)
    param -= alpha*(m_h/(np.sqrt(v_h)+tol))
    return param, m, v


start = time.time()
# 中間層のユニット数とパラメータの初期値
"""
parameter
z1 = wx         (q,)
z2 = y = vz1    (m,)
"""
q = 200
w = np.random.normal(0, 0.3, size=(q, d+1))
v = np.random.normal(0, 0.3, size=(m, q+1))

m_hid = np.zeros(shape=(q, d+1))
v_hid = np.zeros(shape=(q, d+1))

m0 = np.zeros(shape=(m, q+1))
v0 = np.zeros(shape=(m, q+1))


# # option to load weight from file (if file exist)
# w = np.load('w_weight.npy')
# v = np.load('v_weight.npy')

# 確率的勾配降下法によるパラメータ推定
e = []
e_test = []
error = []
error_test = []

num_epoch = 10

eta = 0.1

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)

    eta_t = eta/(epoch + 1)
    for i in index:
        xi = np.append(1, x_train[i, :])
        yi = y_train[i, :]

        # 課題2 ここから
        # 順伝播
        z1, u1 = forward(xi, w, sigmoid)
        z2 = softmax(np.dot(v, z1))

        # 誤差評価
        e.append(CrossEntoropy(z2, yi))

        # 逆伝播
        delta = z2 - yi
        derivative = u1
        V = v[:, 1:]
        delta2 = backward(V, delta, derivative)
        # パラメータの更新
        # v -= eta_t*np.outer(delta, z1)
        # w -= eta_t*np.outer(delta2, xi)
        # update parameter with adam
        grad_v = np.outer(delta, z1)
        grad_w = np.outer(delta2, xi)
        v,m0,v0 = adam(v, m0, v0, grad_v, epoch)
        w,m_hid,v_hid = adam(w, m_hid, v_hid, grad_w, epoch)

        # ここまで

    # training error
    error.append(sum(e)/n)
    print("epoch {} | err {}".format(epoch, sum(e)/n))
    e = []

    # test error
    for j in range(0, n_test):
        xi = np.append(1, x_test[j, :])
        yi = y_test[j, :]

        z1, u1 = forward(xi, w, sigmoid)
        z2 = softmax(np.dot(v, z1))

        e_test.append(CrossEntoropy(z2, yi))

    error_test.append(sum(e_test)/n_test)
    print("        | test err {}".format(sum(e_test)/n_test))
    e_test = []
end = time.time()
print("実行時間： {}".format(end-start))
# 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)  # 青線
plt.plot(error_test, label="test", lw=3)  # オレンジ線
plt.grid()
plt.legend(fontsize=16)
plt.savefig("./error.pdf")

# save weight
np.save('v_weight.npy', v)
np.save('w_weight.npy', w)

# 確率が高いクラスにデータを分類
# モデルの出力を評価
prob = []
for j in range(0, n_test):
    xi = np.append(1, x_test[j, :])
    yi = y_test[j, :]

    z1, u1 = forward(xi, w, sigmoid)
    z2 = softmax(np.dot(v, z1))

    prob.append(z2)

predict = np.argmax(prob, 1)

# 課題3
# confusion matrixを完成させる
ConfMat = np.zeros((m, m))

# plt.clf()
# fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
# fig.show()
# sns.heatmap(ConfMat.astype(dtype=int), linewidths=1,
#             annot=True, fmt="1", cbar=False, cmap="Blues")
# ax.set_xlabel(xlabel="Predict", fontsize=18)
# ax.set_ylabel(ylabel="True", fontsize=18)
# plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)

ConfErr = 0
ConfTrue = 0
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
                ConfMat[i][j] += 1
                ConfErr += 1
        else:
            idx_predict = (predict == j)
            for l in np.where(idx_true*idx_predict == True)[0]:
                ConfMat[i][j] += 1
                ConfTrue += 1

plt.clf()
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype=int), linewidths=1,
            annot=True, fmt="1", cbar=False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)
print("精度： {}%".format(ConfTrue/(ConfTrue+ConfErr)*100))
