### VAE ###
## variational autoencoder ##
# works on principle of encoding data onto a mathematical distribution of values
# derived from the autoencoder necks statistical distribution
# performance increases with the width of the layers leading into the neck increasing #

from keras import backend as K

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon






# testing #
from DataGather import mnist as mn
from Model_Maker import modeler as mm
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers

input_shape = (mn.original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 75




'''
layers = [[Dense,(512),{'activation':'relu'}],
          [Dense,(256),{'activation':'relu'}],
          [Dense,(128),{'activation':None}]]

encoder = mm.make(layers,{'shape':input_shape})
encoder.summary()

z_in = Input(shape=input_shape)
z_e = encoder(z_in)
z_mean = Dense(latent_dim, name='z_mean')(z_e)
z_log_var = Dense(latent_dim, name='z_log_var')(z_e)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
z_encoder = Model(inputs=z_in, outputs=[z_mean, z_log_var, z], name='z_encoder')
z_encoder.summary()

l2 = layers[::-1]
l2.append([Dense,(784),{'activation':'sigmoid'}])
decoder = mm.make(l2,{'shape':(latent_dim,)})
decoder.summary()

v_in = Input(shape=input_shape)
v_encoder = z_encoder(v_in)
v_out = decoder(v_encoder[2])# get z, ignore other outputs
vae = Model(v_in,v_out)
vae.summary()


'''
inputs = Input(shape=input_shape, name='encoder_input')

layers = [[Dense,(1024),{'activation':'relu'}],
          [Dense,(512),{'activation':'relu'}],
          [Dense,(256),{'activation':'relu'}],
          #[Dense,(128),{'activation':'relu'}],
          #[Dense,(64),{'activation':'relu'}]
          ]
lvx = mm.make(layers,{'shape':input_shape})

z_e = lvx(inputs)
z_mean = Dense(latent_dim, name='z_mean')(z_e)
z_log_var = Dense(latent_dim, name='z_log_var')(z_e)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
layers = [#[Dense,(64),{'activation':'relu'}],
          #[Dense,(128),{'activation':'relu'}],
          [Dense,(256),{'activation':'relu'}],
          [Dense,(512),{'activation':'relu'}],
          [Dense,(1024),{'activation':'relu'}],
          [Dense,(mn.original_dim),{'activation':'sigmoid'}]]
lvy = mm.make(layers,{'shape':(latent_dim,)})
z_e = lvy(latent_inputs)
outputs = Dense(mn.original_dim, activation='sigmoid')(z_e)

decoder = Model(latent_inputs, outputs, name='decoder')

models = (encoder, decoder)
data = (mn.x_test, mn.y_test)

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

reconstruction_loss = mse(inputs,outputs) #(inputs, outputs)
# reconstruction_loss = binary_crossentropy(inputs, outputs)

reconstruction_loss *= mn.original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')


vae.fit(mn.x_train,epochs=epochs,batch_size=batch_size,validation_data=(mn.x_test, None))


n,digit_size = 45,28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x,grid_y = np.linspace(-4, 4, n), np.linspace(-4, 4, n)[::-1]

# get predictions across latent space #
z_mean, _, _ = encoder.predict(mn.x_train, batch_size=128)

# graph predictions #
plt.figure(figsize=(12, 10))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=mn.y_train)
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()


# show images across latent space #
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = (n - 1) * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.imshow(figure, cmap='Greys_r')
plt.show()