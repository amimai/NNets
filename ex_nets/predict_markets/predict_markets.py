import pandas as pd
import numpy as np
from matplotlib import pyplot  as plt

from DataProcess import wrangle as wr
from DataProcess import datasets as ds
from DataProcess import display as disp

from Model_Maker import modeler as mm

from keras.layers import Dense, Input, Concatenate
from keras import optimizers, Model



# get our massive dataset for 72 instruments #
data = pd.read_csv('all_data_223k_3y_m5.csv', index_col='date')

# look at data # comented for speed
opens = wr.get_cols(data,['bidopen'])
# disp.show(data,opens)

# several instruments have incomplete datasets for the last 3 years
bad_instruments = ['FRA40', 'CHN50','US2000', 'USOil', 'SOYF', 'WHEATF', 'CORNF', 'EMBasket', 'JPYBasket', 'BTC/USD', 'BCH/USD', 'ETH/USD', 'LTC/USD', 'XRP/USD','CryptoMajor', 'USEquities']
bad_cols = wr.get_cols(data,bad_instruments)

# clean up our data and fill the gaps that are left (from market shutdown over weekend)
data = data.drop(bad_cols, axis=1)
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')

# get the precentage differance of the data
data = wr.p_diff(data)

# mean norm data
data = wr.mean_normalize(data)

# get datasets
data = ds.to_dataset(data)
train1, test1, val1 = ds.random_TTV(data[:-1],data[1:],0.1,0.1)
train2, test2, val2 = ds.random_TTV(data[:-2],data[2:],0.1,0.1)
train3, test3, val3 = ds.random_TTV(data[:-3],data[3:],0.1,0.1)



# # build model


# build fit generator #



# sample test area

def iter_test(t_x,t_y,v_x,v_y,test_x,test_y,
              list_optimise, list_callbacks,list_layers,
              list_batches, list_epoh,verbose=2,
              return_model = False, return_pred = False, return_history=False):
    data = []
    for opt in list_optimise:
        for calls in list_callbacks:
            for epoch in list_epoh:
                for batch in list_batches:
                    for layers in list_layers:
                        model = mm.make(layers, {'shape':(280,)})
                        model.compile(optimizer=opt, loss='mean_squared_error')
                        model.fit(t_x,t_y,validation_data=(v_x,v_y),batch_size=batch,epochs=epoch,callbacks=calls,verbose=verbose)

                        test_dat = []
                        test_dat.append(model.evaluate(test_x,test_y))
                        if return_history:
                            test_dat.append(model.history.history['loss'])
                            test_dat.append(model.history.history['val_loss'])
                        if return_pred:
                            test_dat.append(model.predict(test_x))
                        if return_model:
                            test_dat.append((model))
                        test_dat.append((epoch,batch,calls,opt))
                        data.append(test_dat)
    return data

list_optimise = [optimizers.adam(lr=1e-3, beta_2=.85, epsilon=1e-4)]
list_callbacks = [None]

list_epoh = [20]
list_batches = [156]

list_layers = [
    [[Dense, (1024), {'activation': 'relu'}],
     [Dense, (1024), {'activation': 'relu'}],
     [Dense, (280), {'activation': 'linear'}]]
]
# test some variables and models to get best #
'''
model_test = iter_test(train1[0],train1[1],val1[0],val1[1],test1[0],test1[1],
              list_optimise, list_callbacks,list_layers,
              list_batches, list_epoh,verbose=2,
              return_model = False, return_pred = False, return_history=False)
#model_test.sort()
for i in model_test: print(i)
'''
'''
outputs = p.iter_test(p.train1[0],p.train1[1],p.val1[0],p.val1[1],p.test1[0],p.test1[1],
              p.list_optimise, p.list_callbacks,
              p.list_batches, p.list_epoh,verbose=1,
              return_model = False, return_pred = False, return_history=False)
'''

# iterative long range prediction model #
def iter_build(backprop,data,list_layers,list_optimise,list_batches,list_epoh,list_callbacks,verbose=2):
    datasets = [[0]]
    models = [[0]]
    for i in range(1,backprop):
        datasets.append(ds.seq_stateful(data[:-i],data[i:],0.1,0.1))
        models.append(mm.make(list_layers[0][:-1], {'shape': (280,)}))
        train_wrapper = mm.make([[models[i],None,None],list_layers[0][-1]], {'shape': (280,)})
        train_wrapper.compile(optimizer=list_optimise[0], loss='mean_squared_error')
        train_wrapper.fit(datasets[i][0][0], datasets[i][0][1], validation_data=(datasets[i][1][0], datasets[i][1][1]),
                          batch_size=list_batches[0], epochs=list_epoh[0], callbacks=list_callbacks[0],verbose=verbose)
    inputs = [[0]]
    layers = []
    for i in range(1,backprop):
        inputs.append(Input(shape=(280,)))
        layers.append(models[i](inputs[i]))
    concat = Concatenate()(layers)
    concat2 = Concatenate()(inputs[1:])
    concat3 = Concatenate()([concat,concat2])
    concat3 = Dense(4096,activation='relu')(concat3)
    concat3 = Dense(4096, activation='relu')(concat3)
    outputs = [[0]]
    outnet = [[Dense, (1024), {'activation': 'relu'}],
               [Dense, (1024), {'activation': 'relu'}],
               [Dense, (280), {'activation': 'linear'}]]
    for i in range(1, backprop):
        outputs.append(mm.make(outnet, {'shape': (1024,)})(concat3))
    final_model = Model(inputs=inputs[1:],outputs=outputs[1:])
    final_model.compile(optimizer=list_optimise[0], loss='mean_squared_error')
    final_dat = datasets[1:]
    final_len = len(final_dat[-1][0])-5,len(final_dat[-1][1])-5,len(final_dat[-1][2])-5
    for i in final_dat:
        i = [(i[0][:final_len[0]],i[1][:final_len[0]]),
             (i[0][:final_len[1]],i[1][:final_len[1]]),
             (i[0][:final_len[2]],i[1][:final_len[2]])]
    t_x = []
    t_y = []
    v_x = []
    v_y = []
    for i in final_dat:
        t_x.append(i[0][0][:178420])
        t_y.append(i[0][1][:178420])
        v_x.append(i[1][0][:final_len[1]])
        v_y.append(i[1][1][:final_len[1]])
    final_model.fit(t_x,t_y,validation_data=(v_x,v_y),batch_size=128, epochs=list_epoh[0], callbacks=list_callbacks[0],verbose=verbose)
    return final_model

final_mod = iter_build(12,data,list_layers,list_optimise,list_batches,list_epoh,list_callbacks,verbose=2)