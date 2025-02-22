import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Parameters
DATASET_PATH = "./dataset"
MODEL_PATH = "model.h5"
IMG_SIZE = (128, 128)
CHANNELS = 3
BATCH_SIZE = 128
Z_DIM = 100
EPOCHS = 3

# Load the pre-trained discriminator model
discriminator = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': tf.keras.layers.Layer})
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

# Build Generator
def build_generator():
    model = Sequential([
        layers.Dense(28, bias_regularizer=regularizers.l2(1e-4), input_shape=(Z_DIM,)),
        layers.LeakyReLU(alpha=0.01),
        layers.Dense(28 * 28, activation="tanh", bias_regularizer=regularizers.l2(1e-4)),
        layers.Flatten(),
        layers.Reshape((*IMG_SIZE, CHANNELS))
    ])
    return model

# Build GAN
def build_gan(generator, discriminator):
    model = Sequential([generator, discriminator])
    return model

# Load images and preprocess
def load_images(dataset_path):
    print("[INFO] Loading dataset...")
    image_paths = glob(os.path.join(dataset_path, "*/*"))
    images = []
    labels = []

    for img_path in image_paths:
        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img) / 255.0  # Normalize
        images.append(img)
        labels.append(os.path.basename(os.path.dirname(img_path)))

    print(f"[INFO] Loaded {len(images)} images from dataset.")
    return np.array(images), np.array(labels)

# Train GAN
def train_gan(generator, gan_model, epochs, batch_size):
    print("[INFO] Training GAN...")

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, Z_DIM))
        fake_images = generator.predict(noise)

        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, Z_DIM))
        g_loss = gan_model.train_on_batch(noise, real_labels)

        print(f"D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

# Display generated images
def show_generated_images(generator, num_images=9):
    noise = np.random.normal(0, 1, (num_images, Z_DIM))
    generated_images = generator.predict(noise)

    fig, axs = plt.subplots(3, 3, figsize=(6, 6))
    count = 0

    for i in range(3):
        for j in range(3):
            axs[i, j].imshow(generated_images[count].reshape(IMG_SIZE), cmap="gray")
            axs[i, j].axis("off")
            count += 1

    plt.show()

# Run the pipeline
x_train, y_train = load_images(DATASET_PATH)

generator = build_generator()
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss="binary_crossentropy", optimizer=Adam())

train_gan(generator, gan, EPOCHS, BATCH_SIZE)
show_generated_images(generator)
