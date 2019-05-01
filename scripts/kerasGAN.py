# References https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py


#Setup
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread
from scipy.misc import imresize, imsave

batch_size = 4
noise_size = 256
x_dim = 256
y_dim = 256
num_channels = 3
learning_rate_discriminator = 0.0001
learning_rate_generator = 0.0002
opt = Adam(learning_rate_discriminator, 0.5)

dropout = 0.5
is_training = True

newdir = "chuckResized256"

filepaths_new = []
for dir, _, files in os.walk(newdir):
    for filename in files:
        if not filename.endswith(".jpg"):
            continue
        relDir = os.path.relpath(dir, newdir)
        relFile = os.path.join(relDir, filename)
        filepaths_new.append(newdir + "/" + relFile)

def next_batch(num=64, data=filepaths_new):
    try:
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [imread(data[i]) for i in idx]
        shuffled = np.asarray(data_shuffle)
    except:
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [imread(data[i]) for i in idx]
        shuffled = np.asarray(data_shuffle)
    return shuffled

# Code by Parag Mital (https://github.com/pkmital/CADL/)
def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(
            images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m

def leakyRelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))


def build_discriminator(reuse=None, dropout=dropout):
    momentum = 0.8
    with tf.variable_scope("discriminator", reuse=reuse):
        model = Sequential()

        model.add(Conv2D(32, input_shape=(x_dim, y_dim, num_channels), kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(64, kernel_size=3, strides=2))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=(256, 256, 3))
        validity = model(img)

        return Model(img, validity)


def build_generator(dropout = dropout, reuse=None):
    momentum = 0.8
    with tf.variable_scope("generator", reuse = reuse):
        model = Sequential()

        model.add(Dense(32*32*3, activation='relu', input_dim=(noise_size)))
        model.add(Reshape((32, 32, 3)))
        model.add(UpSampling2D())

        model.add(Conv2D(256, kernel_size = 3, padding='same'))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        model.add(Conv2D(128, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))

        model.add(Conv2D(3, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(noise_size,))
        img = model(noise)

        return Model(noise, img)


def train(generator, discriminator, combined, epochs, batch_size=64, save_interval=100):

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        batch = next_batch(num=batch_size)

        noise = np.random.normal(0, 1, (batch_size, noise_size))
        generated_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(batch, valid)
        d_loss_fake = discriminator.train_on_batch(generated_imgs, fake)
        d_loss = 0.5 * (np.add(d_loss_real, d_loss_fake))

        g_loss = combined.train_on_batch(noise, valid)

        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        if not epoch % save_interval:
            save_images(epoch, generator)


def save_images(epoch, generator):

    rows, cols = 3, 3
    noise = np.random.normal(0, 1, (rows*cols, noise_size))

    generated_imgs = generator.predict(noise)

    #rescaling?
    generated_imgs = 0.5 * generated_imgs + 0.5

    fig, axs = plt.subplots(rows, cols)
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(generated_imgs[cnt, :,:,0])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("generatedImgs/keras%d.png" % epoch)
    plt.close()


discriminator = build_discriminator(dropout=0.8)
generator = build_generator(dropout=0.8)

discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

noiseZ = Input(shape=(noise_size,))
img = generator(noiseZ)

#disabling training the discriminator
discriminator.trainable = False

validity = discriminator(img)

combinedModel = Model(noiseZ, validity)
combinedModel.compile(loss='binary_crossentropy', optimizer=opt)

train(generator, discriminator, combinedModel, epochs=5000, batch_size=batch_size, save_interval=10)
