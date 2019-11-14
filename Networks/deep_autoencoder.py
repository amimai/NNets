### deep_autoencoder ###
## autoencoder using deep connected nets ##

from keras.models import Model, Input
from keras.layers import Dense
from keras import optimizers

from Networks import deeply_connected as dc

import numpy as np

# builds autoencoder based on data stacks #
def model_autoencoder(data,truth,neck,
                      deep_stacks=[[dc.deepcon_dense, dc.core, 3, 80],
                                   [dc.deepcon_dense, dc.core, 4, 70],
                                   [dc.deepcon_dense, dc.core, 5, 60]],
                      pool_stacks=[[Dense, 600, 'relu'],
                                   [Dense, 300, 'relu'],
                                   [Dense, 100, 'relu']]):
    input = Input(shape = data.shape[1:])
    encode = deep_stacks[0][0](input,deep_stacks[0][1],deep_stacks[0][2],deep_stacks[0][3])
    encode = pool_stacks[0][0](pool_stacks[0][1],activation=pool_stacks[0][2])(encode)
    for i in range(1,len(deep_stacks)):
        encode = deep_stacks[i][0](encode, deep_stacks[i][1], deep_stacks[i][2], deep_stacks[i][3])
        encode = pool_stacks[i][0](pool_stacks[i][1],activation=pool_stacks[i][2])(encode)
    encode = Dense(neck, activation='sigmoid')(encode)
    decode = pool_stacks[-1][0](pool_stacks[-1][1],activation=pool_stacks[-1][2])(encode)
    decode = deep_stacks[-1][0](decode,deep_stacks[-1][1],deep_stacks[-1][2],deep_stacks[-1][3])
    for i in range(1,len(deep_stacks)-1):
        decode = pool_stacks[i][0](pool_stacks[i][1],activation=pool_stacks[i][2])(decode)
        decode = deep_stacks[i][0](decode,deep_stacks[i][1],deep_stacks[i][2],deep_stacks[i][3])
    decode =pool_stacks[0][0](pool_stacks[0][1],activation=pool_stacks[0][2])(decode)
    decode = deep_stacks[0][0](decode,deep_stacks[0][1],deep_stacks[0][2],deep_stacks[0][3])

    decode = Dense(truth.shape[-1], activation='linear')(decode)
    model = Model(inputs=input, outputs=decode)
    # create encoders #
    encoder = Model(inputs=input, outputs=encode)
    #encoder.summary()
    # create decoders #
    ## use load model, slicing does not work for complex models ##
    #decoder.load_weights()
    decoder_in = Input(shape=(neck,))
    decoder = pool_stacks[-1][0](pool_stacks[-1][1], activation=pool_stacks[-1][2])(decoder_in)
    decoder = deep_stacks[-1][0](decoder, deep_stacks[-1][1], deep_stacks[-1][2], deep_stacks[-1][3])
    for i in range(len(deep_stacks) - 1):
        decoder = pool_stacks[i][0](pool_stacks[i][1], activation=pool_stacks[i][2])(decoder)
        decoder = deep_stacks[i][0](decoder, deep_stacks[i][1], deep_stacks[i][2], deep_stacks[i][3])
    decoder = pool_stacks[i][0](pool_stacks[i][1], activation=pool_stacks[i][2])(decoder)
    decoder = deep_stacks[0][0](decoder, deep_stacks[0][1], deep_stacks[0][2], deep_stacks[0][3])

    decoder = Dense(truth.shape[-1], activation='linear')(decoder)
    decoder = Model(inputs=decoder_in,outputs=decoder)

    return model, encoder, decoder

# train model #
def train(model,t_x,t_y,v_x,v_y,epochs,verbose,batch_size,
          optimizer=optimizers.adam(lr=1e-3, beta_2=.98, epsilon=1e-5),
          callbacks=None):
    optimizer = optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(t_x, t_y,validation_data=(v_x, v_y), epochs=epochs,
              verbose=verbose, batch_size=batch_size, callbacks=callbacks)


A = np.random.randint(50, size=(1000,5))
B = np.random.randint(50, size=(100,5))
C = np.random.randint(50, size=(100,5))

list_optimise = [optimizers.adam(lr=1e-3, beta_2=.98, epsilon=1e-5),
                 optimizers.sgd(learning_rate=0.001),
                 optimizers.sgd(learning_rate=0.01),
                 optimizers.sgd(learning_rate=0.1),
                 optimizers.sgd(learning_rate=0.1, momentum=0.9, nesterov=True),
                 optimizers.sgd(learning_rate=0.1, momentum=0.99, nesterov=True),
                 optimizers.sgd(learning_rate=0.01, momentum=0.9, nesterov=True),
                 optimizers.sgd(learning_rate=0.01, momentum=0.99, nesterov=True),
                 optimizers.sgd(learning_rate=0.001, momentum=0.9, nesterov=True),
                 optimizers.sgd(learning_rate=0.001, momentum=0.99, nesterov=True)]
list_callbacks = [None]

list_top = [100,50,25]
list_neck = [15,10,5]
list_depth = [4,3,2]

list_epoh = [50,20]
list_batches = [100,10]