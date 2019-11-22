import random
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot  as plt

from DataProcess import wrangle as wr
from DataProcess import datasets as ds
from DataProcess import display as disp

from Model_Maker import modeler as mm

from keras.layers import Dense, Input, Concatenate, Flatten, Reshape
from keras import optimizers, Model

from keras.utils import Sequence

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

# mean norm data (used by generator)
d_mean = data.mean()
d_std = data.std()

# get datasets
#data = ds.to_dataset(data)
#train, test, val = ds.random_TTV(data[:-1],data[1:],0.1,0.1)

backward=12
forward=12

batch_size = 128
epoch = 6

nodes = 512


# # build model
layers = [[Dense, (nodes), {'activation': 'relu'}],
           [Dense, (nodes), {'activation': 'relu'}],
           [Dense, (nodes), {'activation': 'relu'}]]

mods = [mm.make(layers,{'shape':(backward*data.shape[-1],)}),
        mm.make(layers,{'shape':(backward*data.shape[-1]+nodes,)}),
        mm.make(layers,{'shape':(backward*data.shape[-1]+nodes*2,)})]
m_lays_i = Input(shape=(backward*data.shape[-1],))
m_lays1 = mods[0](m_lays_i)
m_lays = Concatenate()([m_lays_i,m_lays1])
m_lays2 = mods[1](m_lays)
m_lays = Concatenate()([m_lays_i,m_lays1,m_lays2])
m_lays3 = mods[2](m_lays)
m_ini = Model(inputs=m_lays_i,outputs=m_lays3)

input = Input(shape=(backward,data.shape[-1]))
l_1 = Flatten()(input)
l_2 = m_ini(l_1)
l_3 = Dense(forward*data.shape[-1],activation='linear')(l_2)
out = Reshape((forward,data.shape[-1]), input_shape=(forward*data.shape[-1],))(l_3)
model = Model(inputs=input,outputs=out)
model.summary()

# build fit generator, returns series #
class data_generator(Sequence):

    def __init__(self, data, batch_size, d_mean=1, d_std=1, backward=12, forward=12, mode="train", aug=None):
        self.data = data
        self.batch_size = batch_size
        self.d_mean =d_mean
        self.d_std =d_std
        self.backward=backward
        self.forward=forward
        self.mode=mode
        self.aug=aug
        self.dat_len = len(data)
        self.avail = []
        self.enum_set()

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def enum_set(self):
        self.avail.extend(random.sample(range(self.backward, self.dat_len - self.forward),
                                        self.dat_len - self.backward - self.forward))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            ran = len(self.avail)
            if ran < 3:
                self.enum_set()
            if self.mode == "eval" and ran<self.batch_size:
                break
            rval = random.randint(0, ran - 1)
            choose = self.avail[rval]
            self.avail.pop(rval)
            dat = np.array((data[choose - self.backward:choose] - self.d_mean) / self.d_std)
            tru = np.array((data[choose:choose + self.forward] - self.d_mean) / self.d_std)
            batch_x.append(np.array(dat))
            batch_y.append(np.array(tru))
        return np.array(batch_x), np.array(batch_y)


def series_generator(data, bs, d_mean=1, d_std=1, backward=12, forward=12, mode="train", aug=None):
    # loop indefinitely
    while True:
        # initialize our batches of images and labels
        dataset = []
        truth = []
        dat_len = len(data)

        # enumerate available items so that full dataset may be searched
        def enum_set(dat_len, backward, forward):
            return random.sample(range(backward, dat_len - forward), dat_len - backward - forward)

        # create the list of available items
        avail = []
        avail.extend(enum_set(dat_len, backward, forward))

        # keep looping until we reach our batch size
        while len(dataset) < bs:

            # maintain our choice list
            ran = len(avail)
            if ran < 3:
                avail.extend(enum_set(dat_len, backward, forward))
                ran=3


            # check to see if the line is empty, indicating we have
            # reached the end of the file
            if mode == "eval" and ran<bs:
                break

            #choose our sample
            rval=random.randint(0,ran-1)
            choose = avail[rval]
            # remove from availability
            avail.pop(rval)

            # extract the dataset and truth set
            dat = np.array((data[choose-backward:choose]-d_mean)/d_std)
            tru = np.array((data[choose:choose+forward]-d_mean)/d_std)


            # update our corresponding batches lists
            dataset.append(np.array(dat))
            truth.append(np.array(tru))

            # if the data augmentation object is not None, apply it
            if aug is not None:
                (dataset, truth) = next(aug.flow(np.array(dat),
                                                 tru, batch_size=bs))

            # yield the batch to the calling function
            yield np.array(dataset), np.array(truth)


# test model
#'''
train_gen = series_generator(data[:160000], batch_size, d_mean=d_mean, d_std=d_std,
                             backward=backward, forward=forward, mode="train", aug=None)
test_gen = series_generator(data[160000:], batch_size, d_mean=d_mean, d_std=d_std,
                             backward=backward, forward=forward, mode="eval", aug=None)
'''
train_gen = data_generator(data[150000:160000], batch_size, d_mean=d_mean, d_std=d_std,
                             backward=backward, forward=forward, mode="train", aug=None)
test_gen = data_generator(data[160000:170000], batch_size, d_mean=d_mean, d_std=d_std,
                             backward=backward, forward=forward, mode="train", aug=None)
#'''

def train(model):
    model.compile(optimizer=optimizers.adam(lr=5e-4, beta_2=.85, epsilon=1e-4), loss='mean_squared_error')
    hist1 = model.fit_generator(train_gen, steps_per_epoch=data.shape[0] // batch_size,
                               validation_data=test_gen, validation_steps=data.shape[0] // batch_size // 10,
                               epochs=epoch, use_multiprocessing=False, workers=1, verbose=2)
    model.compile(optimizer=optimizers.sgd(learning_rate=0.01, momentum=0.99, nesterov=True), loss='mean_squared_error')
    hist2 = model.fit_generator(train_gen, steps_per_epoch=data.shape[0] // batch_size,
                               validation_data=test_gen, validation_steps=data.shape[0] // batch_size // 10,
                               epochs=epoch, use_multiprocessing=False, workers=1, verbose=2)
    return (hist1,hist2)

hist1 = train(model)
hist2 = [hist1]
now = time.time()
completed_models = [model]
models = [m_ini]

while time.time()-now<60*60*7:
    input = Input(shape=(backward, data.shape[-1]))
    l_1 = Flatten()(input)

    mods = [mm.make(layers, {'shape': (backward * data.shape[-1],)}),
            mm.make(layers, {'shape': (backward * data.shape[-1] + nodes,)}),
            mm.make(layers, {'shape': (backward * data.shape[-1] + nodes * 2,)})]
    m_lays_i = Input(shape=(backward * data.shape[-1],))
    m_lays1 = mods[0](m_lays_i)
    m_lays = Concatenate()([m_lays_i, m_lays1])
    m_lays2 = mods[1](m_lays)
    m_lays = Concatenate()([m_lays_i, m_lays1, m_lays2])
    m_lays3 = mods[2](m_lays)
    m_ini_x = Model(inputs=m_lays_i, outputs=m_lays3)

    models.append(m_ini_x)
    l_2 = []
    for mods in models:
        l_2.append(mods(l_1))
    concat = Concatenate()(l_2)
    l_3 = Dense(forward * data.shape[-1], activation='linear')(concat)
    out = Reshape((forward, data.shape[-1]), input_shape=(forward * data.shape[-1],))(l_3)
    model = Model(inputs=input, outputs=out)
    #model.summary()
    print('training model with {0} nodes'.format(len(models)))
    hist2.append(train(model))
    completed_models.append(model)


#'''



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

list_optimise = [optimizers.sgd(learning_rate=0.01, momentum=0.99, nesterov=True)]#[optimizers.adam(lr=5e-3, beta_2=.85, epsilon=1e-4)]
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