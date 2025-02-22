# kerasGANv9.py
# References https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

# Setup
import tensorflow as tf
import time
# No need for session management in TensorFlow 2

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import BinaryCrossentropy

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread
from PIL import Image
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MLflow
import mlflow
import mlflow.keras # For logging Keras models (if needed later)

# mlflow.set_tracking_uri('http://localhost:5000')

batch_size = 4
noise_size = 4
x_dim = 64
y_dim = 64
ndf = 40
ngf=160
num_channels = 3
learning_rate_discriminator = 0.0001
learning_rate_generator = 0.0002
opt = Adam(learning_rate_discriminator, 0.5) # Use tensorflow.keras.optimizers.Adam

dropout = 0.25
is_training = True

newdir = "./dataset"

filepaths_new = []
for dir, _, files in os.walk(newdir):
    for filename in files:
        if not filename.endswith(".jpg") and not filename.endswith(".png") and not filename.endswith(".jpeg"): # Added more image extensions
            continue
        relDir = os.path.relpath(dir, newdir)
        filepaths_new.append(os.path.join(newdir, filename))


def next_batch(data: List[str], num: int, target_size: tuple = (64, 64)) -> Optional[np.ndarray]:
    """
    Load and process a batch of images with proper error handling.
    
    Args:
        data: List of image file paths
        num: Number of images to load
        target_size: Target dimensions for the images
        
    Returns:
        numpy array of processed images or None if batch loading fails
    """
    try:
        # Validate inputs
        if not data or num <= 0:
            raise ValueError("Invalid input parameters")
        
        if num > len(data):
            logger.warning(f"Requested batch size {num} larger than available data {len(data)}")
            num = len(data)

        # Select random indices
        idx = np.random.choice(len(data), size=num, replace=False)
        data_shuffle = []

        for i in idx:
            try:
                # Load and validate each image
                with Image.open(data[i]) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    img_array = np.array(resized_img)
                    
                    # Validate image dimensions
                    if img_array.shape != (*target_size, 3):
                        raise ValueError(f"Invalid image dimensions: {img_array.shape}")
                    
                    data_shuffle.append(img_array)
                    
            except (IOError, OSError) as e:
                logger.error(f"Failed to load image {data[i]}: {str(e)}")
                continue
            except ValueError as e:
                logger.error(f"Invalid image format in {data[i]}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {data[i]}: {str(e)}")
                continue

        # Check if we have any valid images
        if not data_shuffle:
            logger.error("No valid images loaded in batch")
            return None

        # Convert to numpy array and validate final shape
        shuffled = np.array(data_shuffle)
        expected_shape = (len(data_shuffle), *target_size, 3)
        if shuffled.shape != expected_shape:
            raise ValueError(f"Invalid batch shape: {shuffled.shape}, expected {expected_shape}")

        return shuffled

    except Exception as e:
        logger.error(f"Critical error in batch processing: {str(e)}")
        return None


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


def build_discriminator(dropout=dropout):
    momentum = 0.8
    model = Sequential()
    model.add(Conv2D(ndf, input_shape=(x_dim, y_dim, num_channels), kernel_size=4, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(ndf*2, kernel_size=4, padding='same'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(ndf*4, kernel_size=4,padding='same'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(ndf*8, kernel_size=4,padding='same'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, kernel_size=4, padding='same'))
    model.add(Activation('sigmoid'))
    model.summary()
    img = Input(shape=(x_dim, y_dim, num_channels))
    validity = model(img)
    return Model(img, validity)


def build_generator(dropout = dropout):
    momentum = 0.8
    model = Sequential()
    model.add(Conv2DTranspose(ngf*8, input_shape=(None, None, num_channels), kernel_size=4, padding='same'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(ngf*4, kernel_size=4, padding='same'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(ngf*2, kernel_size=4, padding='same'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(ngf, kernel_size=4, padding='same'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(num_channels, kernel_size=4, padding='same'))
    model.add(Activation('tanh'))
    model.summary()
    noise = Input(shape=(None, None, num_channels))
    img = model(noise)
    return Model(noise, img)



def train(generator, discriminator, combinedModel, epochs, batch_size=64, save_interval=100):

    valid = np.ones((batch_size, x_dim, y_dim, 1))
    fake = np.zeros((batch_size, x_dim, y_dim, 1))

    with mlflow.start_run(): # Start MLflow run
        mlflow.log_params({
                "batch_size": batch_size,
                "epochs": epochs,
                "save_interval": save_interval,
                "noise_size": noise_size,
                "x_dim": x_dim,
                "y_dim": y_dim,
                "ndf": ndf,
                "ngf": ngf,
                "learning_rate_discriminator": learning_rate_discriminator,
                "learning_rate_generator": learning_rate_generator,
                "dropout": dropout,
                "optimizer": opt.__class__.__name__
            })

        # Pre-load and cache batches
        cached_batches = [next_batch(num=batch_size) for _ in range(epochs)]

        for epoch in range(epochs):
            epoch_start_time = time.time() # Start time for epoch

            batch = cached_batches[epoch]
            noise = np.random.normal(0.0, 1.0, [batch_size, x_dim, y_dim, num_channels]).astype(np.float32)
            generated_imgs = generator.predict(noise) 

            d_loss_real = discriminator.train_on_batch(batch, valid)
            d_loss_fake = discriminator.train_on_batch(generated_imgs, fake)
            d_loss = 0.5 * (np.add(d_loss_real, d_loss_fake))

            g_loss = combinedModel.train_on_batch(noise, valid)

            epoch_duration = time.time() - epoch_start_time # Calculate epoch duration

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Epoch Time: %.2fs]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, epoch_duration))

            # Log metrics to MLflow
            mlflow.log_metric("discriminator_loss", d_loss[0], step=epoch)
            mlflow.log_metric("discriminator_accuracy", 100*d_loss[1], step=epoch)
            mlflow.log_metric("generator_loss", g_loss, step=epoch)
            mlflow.log_metric("epoch_duration", epoch_duration, step=epoch)


            if not epoch % save_interval:
                save_images(epoch, generator, save_interval=save_interval) # Pass save_interval


def save_images(epoch_num, generator, save_interval=100, sample_size=4):
    """
    Save generated images with improved performance.
    
    Args:
        epoch_num: Current epoch number
        generator: The generator model
        save_interval: How often to save images (in epochs)
        sample_size: Number of images to generate
    """
    # Only save images at specified intervals
    if epoch_num % save_interval != 0:
        return
        
    try:
        # Generate images
        noise = np.random.normal(0.0, 1.0, 
                               [sample_size, x_dim, y_dim, num_channels]
                              ).astype(np.float32)
        
        # Use generator.predict in batch mode for better performance
        generated_imgs = generator.predict(noise, batch_size=sample_size)

        # Create figure
        rows, cols = 2, 2
        fig, axs = plt.subplots(rows, cols, figsize=(8, 8))
        
        cnt = 0
        for i in range(rows):
            for j in range(cols):
                if cnt < len(generated_imgs):
                    # Normalize image data for display
                    img_data = generated_imgs[cnt]
                    img_data = (img_data + 1) * 127.5  # Denormalize if using tanh
                    axs[i, j].imshow(img_data.astype(np.uint8))
                    axs[i, j].axis('off')
                    cnt += 1

        # Save with optimized settings
        save_dir = os.path.join(".", "generatedImgsv2", "keras")
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"epoch_{epoch_num:06d}.png"  # Zero-padded epoch number
        filepath = os.path.join(save_dir, filename)
        
        # Use optimized saving parameters
        plt.savefig(filepath, 
                   bbox_inches="tight",
                   dpi=100,  # Adjust DPI as needed
                   optimize=True,
                   quality=85)  # Slightly reduced quality for better performance
        plt.close(fig)  # Explicitly close figure to free memory

        # Log to MLflow only at save intervals
        if mlflow.active_run():
            mlflow.log_artifact(filepath, "generated_images")
            
    except Exception as e:
        logger.error(f"Error saving images at epoch {epoch_num}: {str(e)}")
        
    finally:
        # Ensure figure is closed even if an error occurs
        plt.close('all')

print("Building discriminator and generator...")
discriminator = build_discriminator(dropout=dropout)
discriminator_loss = BinaryCrossentropy() # Use the object, not string
discriminator.compile(loss=discriminator_loss, optimizer=opt, metrics=['accuracy']) # Use optimizer=opt which is tensorflow.keras.optimizers.Adam
generator = build_generator(dropout=dropout)

print("Generating noiseZ and image...")
noiseZ = Input(shape=(x_dim, y_dim, num_channels)) # Use tensorflow.keras.layers.Input
img = generator(noiseZ)

#disabling training the discriminator
discriminator.trainable = False

validity = discriminator(img)

combinedModel = Model(noiseZ, validity) # Use tensorflow.keras.models.Model
combinedModel.compile(loss='binary_crossentropy', optimizer=opt) # Use optimizer=opt which is tensorflow.keras.optimizers.Adam

print("Training GAN...")
train(generator, discriminator, combinedModel, epochs=50000, batch_size=batch_size, save_interval=5) # combined is renamed to combinedModel