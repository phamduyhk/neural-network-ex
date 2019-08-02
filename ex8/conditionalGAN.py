import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, UpSampling2D
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import time


def G_model(Height, Width, channel=3):
    inputs = Input((100,))
    in_h = int(Height / 4)
    in_w = int(Width / 4)
    x = Dense(in_h * in_w * 128, activation='tanh', name='g_dense1')(inputs)
    x = BatchNormalization()(x)
    x = Reshape((in_h, in_w, 128), input_shape=(128 * in_h * in_w,))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (5, 5), padding='same',
               activation='tanh', name='g_conv1')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(channel, (5, 5), padding='same',
               activation='tanh', name='g_out')(x)
    model = Model(inputs, x, name='G')
    return model


# Discreminatorの定義
def D_model(Height, Width, channel=3):
    inputs = Input((Height, Width, channel))
    x = Conv2D(64, (5, 5), padding='same',
               activation='tanh', name='d_conv1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (5, 5), padding='same',
               activation='tanh', name='d_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='d_dense1')(x)
    x = Dense(1, activation='sigmoid', name='d_out')(x)
    model = Model(inputs, x, name='D')
    return model


def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model

# for output
def save_images(imgs, index, dir_path):
    # Argment
    #  img_batch = np.array((batch, height, width, channel)) with value range [-1, 1]
    B, H, W, C = imgs.shape
    batch = imgs * 127.5 + 127.5
    batch = batch.astype(np.uint8)
    w_num = np.ceil(np.sqrt(B)).astype(np.int)
    h_num = int(np.ceil(B / w_num))
    out = np.zeros((h_num*H, w_num*W), dtype=np.uint8)
    for i in range(B):
        x = i % w_num
        y = i // w_num
        out[y*H:(y+1)*H, x*W:(x+1)*W] = batch[i, ..., 0]
    fname = str(index).zfill(len(str(3000))) + '.jpg'
    save_path = os.path.join(dir_path, fname)

    plt.imshow(out, cmap='gray')
    plt.title("iteration: {}".format(index))
    plt.axis("off")
    plt.savefig(save_path)

if __name__ == '__main__':
    g = G_model(Height=28, Width=28, channel=1)
    d = D_model(Height=28, Width=28, channel=1)
    c = Combined_model(g=g, d=d)

    g_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    d_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

    g.compile(loss='binary_crossentropy', optimizer='SGD')
    d.trainable = False
    for layer in d.layers:
        layer.trainable = False
    c.compile(loss='binary_crossentropy', optimizer=g_opt)
    d.trainable = True
    for layer in d.layers:
        layer.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_opt)

    # MNIST DATASET
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    train_num = X_train.shape[0]
    train_num_per_step = train_num // 64

    # config
    Minibatch = 64
    Save_test_img_dir = "./"
    iteration = 3000
    # trainning
    for ite in range(iteration):
        start = time.time()
        ite += 1
        # Discremenator training
        train_ind = ite % (train_num_per_step - 1)
        y = X_train[train_ind * Minibatch: (train_ind+1) * Minibatch]
        input_noise = np.random.uniform(-1, 1, size=(64, 100))
        g_output = g.predict(input_noise, verbose=0)
        X = np.concatenate((y, g_output))
        Y = [1] * 64 + [0] * 64
        d_loss = d.train_on_batch(X, Y)
        # Generator training
        input_noise = np.random.uniform(-1, 1, size=(Minibatch, 100))
        g_loss = c.train_on_batch(input_noise, [1] * Minibatch)
        end = time.time()
        print("ite {}, d_loss {}, g_loss {}".format(ite,d_loss,g_loss))
    
    save_images(g_output, index=5, dir_path=Save_test_img_dir)

