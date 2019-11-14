### deeply_connected ###
## basic implimentation of https://arxiv.org/pdf/1608.06993.pdf ##

# from keras.models import Sequential, load_model, Model, Input
from keras.layers import Dense, Dropout, Concatenate #, Flatten, Activation, Add, Reshape
from keras.layers import LSTM
from keras.constraints import maxnorm
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
# from keras import optimizers


# method for implementing core for deep connected nets #
def core(input,size):
    layers = Dense(size, activation='relu',
               kernel_initializer='normal', kernel_constraint=maxnorm(3))(input)
    layers = Dropout(rate=0.4)(layers)
    layers = Dense(size, activation='relu',
                   kernel_initializer='normal', kernel_constraint=maxnorm(3))(layers)
    return layers

# implementation of dense #
def deepcon_dense(input,cores,stacks,size):
    BigAg = [input]
    BigAdd = input
    for i in range(stacks):
        stack = cores(BigAdd,size)           # get a core
        BigAg.append(stack)             # append to list of outputs
        BigAdd = Concatenate()(BigAg)   # concatenate outputs for next layer
    return BigAdd