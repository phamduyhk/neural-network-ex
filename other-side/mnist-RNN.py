import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder("float", [None, 28, 28])
y = tf.placeholder("float", [None, 10])


def RNN(x):
    x = tf.unstack(x, 28, 1)

    # LSTMの設定
    lstm_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)

    # モデルの定義。各タイムステップの出力値と状態が返される
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # 重みとバイアスの設定
    weight = tf.Variable(tf.random_normal([128, 10]))
    bias = tf.Variable(tf.random_normal([10]))

    return tf.matmul(outputs[-1], weight) + bias


cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# 評価用
correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


batch_size = 128
n_training_iters = 100000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < n_training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # next_batchで返されるbatch_xは[batch_size, 784]のテンソルなので、batch_size×28×28に変換します。
        batch_x = batch_x.reshape((batch_size, 28, 28))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print('step: {} / loss: {:.6f} / acc: {:.5f}'.format(step, loss, acc))
        step += 1

    # テスト
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, 28, 28))
    test_label = mnist.test.labels[:test_len]
    test_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label})
    print("Test Accuracy: {}".format(test_acc))
