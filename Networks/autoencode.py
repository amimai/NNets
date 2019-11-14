### autoencode ###
## code to build and use autoencoders ##

from keras.models import Model, Input
from keras.layers import Dense
from keras import optimizers

import numpy as np

# builds autoencoder based on data suplied #
def model_autoencoder(data,truth,top=720, neck=50,depth=2):
    input = Input(shape = data.shape[1:])
    encode = Dense(top,activation='relu')(input)
    for i in range(depth-1):
        encode = Dense( int(top-(top-neck)/depth*(i+1)), activation='relu')(encode)
    encode = Dense(neck, activation='sigmoid')(encode)
    decode = Dense(int(neck + (top - neck) / depth * (1)), activation='relu')(encode)
    for i in range(1,depth-1):
        decode = Dense( int(neck+(top-neck)/depth*(i+1)), activation='relu')(decode)
    decode = Dense(top,activation='relu')(decode)
    decode = Dense(truth.shape[-1], activation='linear')(decode)
    model = Model(inputs=input, outputs=decode)
    # create encoders #
    encoder = Model(inputs=input, outputs=encode)
    # create decoders #
    input_enc = Input(shape=(neck,))
    decoder_layers = model.layers[-1-depth](input_enc)
    for each in model.layers[-depth:]:
        decoder_layers = each(decoder_layers)

    decoder = Model(inputs=input_enc,outputs=decoder_layers)
    return model, encoder, decoder

# train model #
def train(model,t_x,t_y,v_x,v_y,epochs,verbose,batch_size,
          optimizer=optimizers.adam(lr=1e-3, beta_2=.98, epsilon=1e-5),
          callbacks=None):
    optimizer = optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(t_x, t_y,validation_data=(v_x, v_y), epochs=epochs,
              verbose=verbose, batch_size=batch_size, callbacks=callbacks)

# iterate over variables #
def iter_test(t_x,t_y,v_x,v_y,test_x,test_y,
              list_optimise, list_callbacks,
              list_batches, list_epoh,
              list_top,list_neck,list_depth,
              return_model = False, return_pred = False, return_history=False):
    data = []
    for opt in list_optimise:
        for calls in list_callbacks:
            for epoch in list_epoh:
                for batch in list_batches:
                    for top in list_top:
                        for neck in list_neck:
                            for depth in list_depth:
                                model, encode, decode = model_autoencoder(t_x, t_y, top=top, neck=neck, depth=depth)
                                train(model,t_x,t_y,v_x,v_y,epochs=epoch,batch_size=batch,
                                      optimizer=opt,callbacks=calls,verbose=2)
                                test_dat = []
                                test_dat.append(model.evaluate(test_x,test_y))
                                if return_history:
                                    test_dat.append(model.history.history['loss'])
                                    test_dat.append(model.history.history['val_loss'])
                                if return_pred:
                                    test_dat.append(model.predict(test_x))
                                if return_model:
                                    test_dat.append((model, encode, decode))
                                test_dat.append((top,neck,depth,epoch,batch,calls,opt))
                                data.append(test_dat)
    return data


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

# ideal config : top:50, neck:4, depth:5, epochs:50, batches:10, adam






