import numpy as np
import time
import misc as mi

from keras.models import Sequential, load_model, Model, Input
from keras.layers import Dense, Dropout, Concatenate, Flatten, Activation, Add, Reshape
from keras.layers import LSTM
from keras.constraints import maxnorm
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers


def model_lstm_seq(data,truth,nodes,layers):
    input = Input(batch_shape=(1,1,data.shape[-1]))
    accordian = []
    layer = LSTM(nodes,return_sequences=True,activation='relu',stateful=True,
                  kernel_initializer='normal',kernel_constraint=maxnorm(3))(input)
    layer = Dropout(rate=0.4)(layer)
    for i in range(layers):
        layer = LSTM(nodes, return_sequences=True, activation='relu',stateful=True,
                     kernel_initializer='normal', kernel_constraint=maxnorm(3))(layer)
        layer = Dropout(rate=0.4)(layer)
    layer = LSTM(nodes, return_sequences=False, activation='relu',stateful=True,
                 kernel_initializer='normal', kernel_constraint=maxnorm(3))(layer)
    layer = Dropout(rate=0.4)(layer)
    activation = Dense(truth.shape[-1], activation='linear')(layer)
    return Model(inputs=input, outputs=activation)


def model_lstm_seq_accord(data,truth,nodes,layers):
    input = Input(batch_shape=(1,1,data.shape[-1]))
    accordian = []
    layer = LSTM(nodes,return_sequences=True,activation='relu',stateful=True,
                  kernel_initializer='normal',kernel_constraint=maxnorm(3))(input)
    layer = Dropout(rate=0.4)(layer)
    accordian.append(LSTM(nodes,return_sequences=False,activation='relu',stateful=True,
                  kernel_initializer='normal',kernel_constraint=maxnorm(3))(layer))
    for i in range(layers):
        layer = LSTM(nodes, return_sequences=True, activation='relu',stateful=True,
                     kernel_initializer='normal', kernel_constraint=maxnorm(3))(layer)
        layer = Dropout(rate=0.4)(layer)
        accordian.append(LSTM(nodes, return_sequences=False, activation='relu',stateful=True,
                              kernel_initializer='normal', kernel_constraint=maxnorm(3))(layer))
    layer = Concatenate()(accordian)
    layer = Dropout(rate=0.4)(layer)
    activation = Dense(truth.shape[-1], activation='linear')(layer)
    return Model(inputs=input, outputs=activation)



def train(model,t_x,t_y,v_x,v_y,epochs,batch_size=1,verbose=2,
          optimizer=optimizers.adam(lr=1e-5, beta_2=.99, epsilon=1e-9),
          callbacks=None):
    optimizer = optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = []
    for i in range(epochs):
        print ('epoch : ', i)
        mean_tr_loss = []
        mean_vl_loss = []
        mi.startProgress('ep:'+str(i)+': ')
        for i in range(len(t_x)):
            tr_loss = model.train_on_batch(t_x[i].reshape(1,1,t_y.shape[-1]),t_y[i].reshape(1,t_y.shape[-1]))
            mi.progress(str(tr_loss))
            mean_tr_loss.append(tr_loss)
        model.reset_states()
        for i in range(len(v_x)):
            vl_loss = model.train_on_batch(v_x[i].reshape(1,1,v_y.shape[-1]),v_y[i].reshape(1,v_y.shape[-1]))
            mean_vl_loss.append(vl_loss)
        model.reset_states()
        mi.endProgress()
        avgs = (np.mean(mean_tr_loss),np.mean(mean_vl_loss))
        history.append( avgs )
        print (avgs)
    return history

