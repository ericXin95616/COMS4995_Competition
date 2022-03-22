from keras.models import Model, Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Flatten, Dense, Reshape
import tensorflow as tf
import numpy as np
import time
from utils import *

'''
Input: 750x1845x3 hazed image -> 768 x 1856 x 3
Loss: ELBO loss between decoded image and clean image
'''
class VAE(Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # define encoder
        self.encoder = Sequential()
        self.encoder.add(InputLayer(input_shape=(768, 1856, 3)))
        self.encoder.add(Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'))
        self.encoder.add(MaxPooling2D((2, 2)))
        self.encoder.add(Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'))
        self.encoder.add(MaxPooling2D((2, 2)))
        self.encoder.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'))
        # Output should be of dim (11, 28, 64)
        self.encoder.add(MaxPooling2D((2, 2)))
        self.encoder.add(Flatten())
        # map to latent space
        self.encoder.add(Dense(2 * latent_dim))
        self.encoder.summary()

        # define decoder
        self.decoder = Sequential()
        self.decoder.add(InputLayer(input_shape=(latent_dim, )))
        self.decoder.add(Dense(units=12*29*64, activation='relu'))
        self.decoder.add(Reshape(target_shape=(12, 29, 64)))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'))
        self.decoder.add(UpSampling2D((2 , 2)))
        self.decoder.add(Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'))
        # No activation
        self.decoder.add(Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same'))
        self.decoder.summary()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        encoded_x = self.encoder(x)
        mean, logvar = tf.split(encoded_x, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def preprocess_image(im):
    assert (750, 1845, 3) == im.shape
    # padding image to (768, 1856, 3)
    old_image_height, old_image_width, channels = im.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = 1856
    new_image_height = 768
    color = (0, 0, 0)
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = im
    # rescaling image from [0, 255] to [0, 1] by dividing 255
    result = np.divide(result.astype(np.float32), 255)
    result.shape = (1, 768, 1856, 3)
    return result


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x, y):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def train_step(model, x, y, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train(model, train_images, val_images, epochs=10, optimizer=tf.keras.optimizers.Adagrad(learning_rate=1e-2)):
    for epoch in range(epochs):
        print('Epoch {} begins'.format(epoch))
        start_time = time.time()
        for train_im_path in train_images:
            hazy_image, clean_image = get_hazy_clean_image(train_im_path)
            # padding image
            hazy_image = preprocess_image(hazy_image)
            clean_image = preprocess_image(clean_image)
            train_step(model, hazy_image, clean_image, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for val_im_path in val_images:
            hazy_image, clean_image = get_hazy_clean_image(val_im_path)
            # padding image
            hazy_image = preprocess_image(hazy_image)
            clean_image = preprocess_image(clean_image)
            loss(compute_loss(model, hazy_image, clean_image))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))
        # save weights
        model.save_weights('./checkpoints/VAE_checkpoint')


def generate_and_save_images(model, test_sample):
    predicts = []
    clean_images = []
    for test in test_sample:
        hazy_image, clean_image = get_hazy_clean_image(test)
        hazy_image = preprocess_image(hazy_image)
        mean, logvar = model.encode(hazy_image)
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
        assert predictions[0].shape == (768, 1856, 3)
        predicts.append(predictions[0])
        clean_images.append(clean_image)
    return predicts, clean_images


# optimizer = tf.keras.optimizers.Adam(1e-4)
epochs = 10
latent_dim = 5000

model = VAE(latent_dim=latent_dim)

# train(model, train_images=train_images, val_images=val_images, epochs=1500)

model.load_weights('./checkpoints/VAE_checkpoint')

predicts, clean_images = generate_and_save_images(model, test_sample=val_images[-2:])

fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.multiply(predicts[0], 255).astype(int))
ax[1].imshow(clean_images[0])
plt.show()

