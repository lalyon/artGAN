# DCGAN
# Source https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/DCGAN-MNIST.ipynb

#Setup
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread
from scipy.misc import imresize, imsave

tf.reset_default_graph()
batch_size = 1
noise_size = 64
x_dim = 256
y_dim = 256
learning_rate_discriminator = 0.0001
learning_rate_generator = 0.0002

X = tf.placeholder(dtype=tf.float32, shape=[None, x_dim, y_dim, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, noise_size])

dropout = tf.placeholder(dtype=tf.float32, name='dropout')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

#file path vars
# parent_dir_1 = "dataset_updated"
# parent_dir_2 = "dataset_updated_2"
# sub_dir_1 = "training_set"
# sub_dir_2 = "validation_set"
# sub_sub_dir_1 = "drawings"
# sub_sub_dir_2 = "engraving"
# sub_sub_dir_3 = "iconography"
# sub_sub_dir_4 = "painting"
# sub_sub_dir_5 = "sculpture"
newdir = "impressionistImages256"

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

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. -x) * tf.log(1. - z + eps)))

def discriminator(input_image, reuse = None, dropout=dropout):
    with tf.variable_scope("discriminator", reuse=reuse):
        # Input layer
        # shape=[-1, ...] infers the number of images
        x = tf.reshape(input_image, shape=[-1, x_dim, y_dim, 3])

        # Convolution #1
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same', activation = leakyRelu)
        # Dropout #1
        x = tf.layers.dropout(x, dropout)
        # Convolution #2
        x= tf.layers.conv2d(x, kernel_size=5, filters = 128, strides=1, padding='same', activation = leakyRelu)
        # Dropout #2
        x = tf.layers.dropout(x, dropout)

        # Convolution #3
        x = tf.layers.conv2d(x, kernel_size=5, filters = 64, strides=1, padding='same', activation = leakyRelu)
        # Dropout #3
        x = tf.layers.dropout(x, dropout)

        # Flatten and Dense to one output
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=64, activation = leakyRelu)
        x = tf.layers.dense(x, units=1, activation = tf.nn.sigmoid)
        return x

def generator(input_noise, dropout = dropout, reuse = None, is_training = is_training):
    momentum = 0.9 #could be tweaked
    with tf.variable_scope("generator", reuse = None):
        x = input_noise
        dimension_one = 4
        dimension_two = 3 #channels

        # Dense #1
        x = tf.layers.dense(x, units=dimension_one * dimension_one * dimension_two, activation=leakyRelu)
        # Dropout #1
        x = tf.layers.dropout(x, dropout)
        # Batch Normalization #1
        x = tf.contrib.layers.batch_norm(x, is_training = is_training, decay = momentum)
        # Reshape #1
        x = tf.reshape(x, shape = [-1, dimension_one, dimension_one, dimension_two])
        # Reshape #2
        x = tf.image.resize_images(x, size=[64,64])

        # Transpose #1
        x = tf.layers.conv2d_transpose(x, kernel_size = 5, filters=256, strides=2, padding='same', activation=leakyRelu)
        # Dropout #2
        x = tf.layers.dropout(x, dropout)
        # Batch Normalization #2
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay = momentum)

        # Transpose #2
        x = tf.layers.conv2d_transpose(x, kernel_size = 5, filters = 128, strides=2, padding = 'same', activation=leakyRelu)
        # Dropout #3
        x = tf.layers.dropout(x, dropout)
        # Batch Normalization #3
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay = momentum)

        # Transpose #3
        x = tf.layers.conv2d_transpose(x, kernel_size = 5, filters = 64, strides=1, padding = 'same', activation=leakyRelu)
        # Dropout #4
        x = tf.layers.dropout(x, dropout)
        # Batch Normalization #4
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay = momentum)

        # Final Transpose #4
        x = tf.layers.conv2d_transpose(x, kernel_size = 5, filters = 3, strides=1, padding = 'same', activation=tf.nn.sigmoid)
        return x

gen = generator(noise, dropout, None, is_training)
print(gen)

discriminator_real = discriminator(X)
print(discriminator_real)
discriminator_fake = discriminator(gen, reuse=True)

variables_generator = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
variables_descriminator = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

regularization_discriminator = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), variables_descriminator)
regularization_generator = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), variables_generator)

loss_discriminator_real = binary_cross_entropy(tf.ones_like(discriminator_real), discriminator_real)
loss_discriminator_fake = binary_cross_entropy(tf.zeros_like(discriminator_fake), discriminator_fake)
loss_generator = tf.reduce_mean(binary_cross_entropy(tf.ones_like(discriminator_fake), discriminator_fake))

loss_discriminator = tf.reduce_mean(0.5 * (loss_discriminator_real + loss_discriminator_fake))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    opt_discrim = tf.train.RMSPropOptimizer(learning_rate=learning_rate_discriminator).minimize(loss_discriminator + regularization_discriminator, var_list=variables_descriminator)
    opt_gen = tf.train.RMSPropOptimizer(learning_rate=learning_rate_generator).minimize(loss_generator + regularization_generator, var_list=variables_generator)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



# Training

num_epochs = 50000


for epoch in range(num_epochs):

    print(epoch)
    training_discriminator = True
    training_generator = True

    dropout_training = 0.3

    n = np.random.uniform(0.0, 1.0, [batch_size, noise_size]).astype(np.float32)
    batch = [b for b in next_batch(num=batch_size)]


    discrim_real_ls, discrim_fake_ls, gen_ls, discrim_ls = sess.run([loss_discriminator_real, loss_discriminator_fake, loss_generator, loss_discriminator], feed_dict={X: batch, noise: n, dropout: dropout_training, is_training:True})

    discrim_fake_loss_init = discrim_fake_ls

    discrim_real_loss = np.mean(discrim_real_ls)
    discrim_fake_loss = np.mean(discrim_fake_ls)

    gen_ls = gen_ls
    discrim_ls = discrim_ls

    if gen_ls * 1.5 < discrim_ls:
        training_generator = False
        pass
    if discrim_ls * 1.5 < gen_ls:
        training_discriminator = False
        pass


    if training_discriminator:
        sess.run(opt_discrim, feed_dict={noise: n, X: batch, dropout: dropout_training, is_training:True})

    if training_generator:
        sess.run(opt_gen, feed_dict={noise: n, dropout: dropout_training, is_training:True})

    if not epoch % 10:
        print(epoch, discrim_ls, gen_ls)
        if not training_generator:
            print("not training generator")
        if not training_discriminator:
            print("not training discriminator")
        generate_images = sess.run(gen, feed_dict = {noise: n, dropout: 1.0, is_training: False})
        images = [img[:,:,:] for img in generate_images]
        mont = montage(images)
        plt.axis('off')
        plt.imshow(mont, cmap='gray')
        savestring = "./generatedImpressionistImages256/art" + str(epoch) + ".png"
        plt.savefig(savestring, bbox_inches="tight")
