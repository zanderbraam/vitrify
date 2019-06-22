import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Get MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_train.shape[1:])))

# Set parameters
batch_size = 250
original_dim = 784
latent_dim = 10
intermediate_dim_1 = 512
intermediate_dim_2 = 256
intermediate_dim_3 = 128
epochs = 100
epsilon_std = 1.0
adam = Adam(lr=0.001)

# Specifying input
x = Input(shape=(original_dim,))

encoder_h_1 = Dense(intermediate_dim_1, activation='relu')(x)
encoder_h_2 = Dense(intermediate_dim_2, activation='relu')(encoder_h_1)
encoder_h_3 = Dense(intermediate_dim_3, activation='relu')(encoder_h_2)
z_mean = Dense(latent_dim)(encoder_h_3)
z_log_var = Dense(latent_dim)(encoder_h_3)


# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Sampling latent space
z = Lambda(sampling)([z_mean, z_log_var])

# Instantiate these layers separately so as to reuse them later
decoder_h_1 = Dense(intermediate_dim_3, activation='relu')
decoder_h_2 = Dense(intermediate_dim_2, activation='relu')
decoder_h_3 = Dense(intermediate_dim_1, activation='relu')
decoder_out = Dense(original_dim, activation='sigmoid')
h_1_decoded = decoder_h_1(z)
h_2_decoded = decoder_h_2(h_1_decoded)
h_3_decoded = decoder_h_3(h_2_decoded)
outputs = decoder_out(h_3_decoded)


# Specifying loss
def vae_loss(x_true, x_pred):
    recon = K.sum(K.square(x_true - x_pred), axis=-1)
    kl = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return recon + kl

# Instantiate models for VAE and encoder
vae = Model(x, outputs)
encoder = Model(x, z_mean)
encoder.summary()

# Instantiate model for decoder
d_z = Input(shape=(latent_dim,))
d_h_1 = decoder_h_1(d_z)
d_h_2 = decoder_h_2(d_h_1)
d_h_3 = decoder_h_3(d_h_2)
d_out = decoder_out(d_h_3)
decoder = Model(d_z, d_out)
decoder.summary()

# Compile
vae.compile(optimizer=adam, loss=vae_loss)

# Train the VAE (or reload a previously trained one)
weights_file = "vae_%d_latent.hdf5" % latent_dim
if os.path.isfile(weights_file):
    vae.load_weights(weights_file)
else:
    from keras.callbacks import History
    hist_cb = History()
    vae.fit(x_train,
            x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')],
            validation_data=(x_test, x_test))
    vae.save_weights(weights_file)

vae.summary()

# Display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
f = plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
# plt.show()
f.savefig("latent_space_vae_128.pdf", bbox_inches='tight')

# Display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

pattern = x_test[3]
digit = pattern.reshape(28, 28)
plt.imshow(digit, cmap='Greys_r')
plt.show()
label = y_test[3]
print(label)

vae_pat = vae.predict(pattern.reshape(1, 784))
digit = vae_pat.reshape(28, 28)
plt.imshow(digit, cmap='Greys_r')
plt.show()
