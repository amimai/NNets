import random
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot  as plt

from DataProcess import wrangle as wr
from DataProcess import datasets as ds
from DataProcess import display as disp

from Model_Maker import modeler as mm

from keras.layers import Dense, Input, Concatenate, Flatten, Reshape, Activation, LSTM
from keras import optimizers, Model

from keras.utils import Sequence

from callbacks.CosHot import CosHotRestart

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

#good_cols = wr.get_cols(data,['EUR/USD'])# ['bidopen','bidhigh','bidlow'])#['bidopen'])#
#data = data[good_cols]
# get the precentage differance of the data
data = wr.p_diff(data)

# mean norm data (used by generator)
d_mean = data.mean()
d_std = data.std()

# get datasets
#data = ds.to_dataset(data)
#train, test, val = ds.random_TTV(data[:-1],data[1:],0.1,0.1)

backward=24
forward=6

batch_size = 128
epoch = 10

nodes = 64

# extract truth data
truth = data.shift(forward).rolling(forward, min_periods=1).sum()
t_mean = truth.mean()
t_std = truth.std()


# # build model

layers = [[LSTM, (nodes), {'activation': 'relu','return_sequences':True}]]

deep1 = mm.make_LSTMdeepnet(layers,(backward,data.shape[-1]),128,5)


#print('dense1 made')
def makeme():
    input_l = Input(shape=(backward,data.shape[-1]))
    l_2 = deep1(input_l)
    l_3 = LSTM(1*data.shape[-1],activation='linear')(l_2)
    out = Reshape((1,data.shape[-1]), input_shape=(1*data.shape[-1],))(l_3)
    model = Model(inputs=input_l,outputs=out)
    return model
    #model.summary()

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


def series_generator(data,true, bs, d_mean=1, d_std=1, t_mean=1,t_std=1, backward=12, forward=12, mode="train", aug=None):
    # loop indefinitely
    while True:
        # initialize our batches of images and labels
        dataset = []
        truth = []
        dat_len = len(data)

        # enumerate available items so that full dataset may be searched
        def enum_set(dat_len, backward, forward):
            samples = random.sample(range(backward, dat_len - forward), dat_len - backward - forward)
            if mode=='pred':
                samples=[]
                for i in range(backward, dat_len - forward):
                    samples.append(i)
            return samples

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
            if mode=='pred': #get first value
                rval = 0
            choose = avail[rval]
            # remove from availability
            avail.pop(rval)

            # extract the dataset and truth set
            dat = np.array((data[choose-backward:choose]-d_mean)/d_std)
            tru = np.array((true[choose:choose+1]-t_mean)/t_std)


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
train_gen = series_generator(data[:160000],truth[:160000], batch_size, d_mean=d_mean, d_std=d_std,t_mean=t_mean, t_std=t_std,
                             backward=backward, forward=forward, mode="train", aug=None)
test_gen = series_generator(data[160000:],truth[160000:], batch_size, d_mean=d_mean, d_std=d_std,t_mean=t_mean, t_std=t_std,
                             backward=backward, forward=forward, mode="eval", aug=None)
pred_gen = series_generator(data,truth, batch_size, d_mean=d_mean, d_std=d_std,t_mean=t_mean, t_std=t_std,
                             backward=backward, forward=forward, mode="pred", aug=None)
'''
train_gen = data_generator(data[150000:160000], batch_size, d_mean=d_mean, d_std=d_std,
                             backward=backward, forward=forward, mode="train", aug=None)
test_gen = data_generator(data[160000:170000], batch_size, d_mean=d_mean, d_std=d_std,
                             backward=backward, forward=forward, mode="train", aug=None)
#'''

#
#optimizers.adam(lr=1e-3, beta_2=.85, epsilon=1e-4)
def train(model,optimizer=optimizers.sgd(learning_rate=0.01, momentum=0.9, nesterov=True),epochs=150,callbacks=None):
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    hist1 = model.fit_generator(train_gen, steps_per_epoch=data.shape[0] // batch_size,
                               validation_data=test_gen, validation_steps=data.shape[0] // batch_size // 10,
                               epochs=epochs, use_multiprocessing=False, workers=1, verbose=2, callbacks=callbacks)
    return (hist1)

hist = []
for i in range(4,5):
    lr = 1*10**-4
    model = makeme()
    hist.append(train(model,optimizer=optimizers.sgd(learning_rate=lr, momentum=0.9, nesterov=True), epochs=epoch,
                      callbacks=[CosHotRestart(nb_epochs=epoch,nb_cycles=epoch//5,Pgain=1.1,LRgain=.8,verbose=2, valweight=False,save_model='callbacks/CosHot_weights/',hammer=0)]
                      ))


if __name__ == '__main__':


    # iterative long range prediction model #

    for i in range(len(hist)):
        plt.figure()
        plt.plot(hist[i].history['loss'])
        plt.plot(hist[i].history['val_loss'])


    pred = model.predict(np.array((data[25-backward:25]-d_mean)/d_std).reshape(1,24,280))
    pred = pd.DataFrame(pred[0])
    pred.columns=truth.columns
    pred = (pred*t_std)+t_mean


    pred_d = []
    pred_t = []
    for i in range(150000,151000):
        pred_d.append(np.array((data[i - backward:i] - d_mean) / d_std))
        pred_t.append(np.array((truth[i:i+1]-t_mean)/t_std))
    pred_r = model.predict(np.array(pred_d))
    pred_r = pd.DataFrame(pred_r.reshape(1000,280))
    pred_t = pd.DataFrame(np.array(pred_t).reshape(1000,280))
    pred_r.columns = truth.columns
    pred_t.columns = truth.columns
    pred_dif = pred_t-pred_r
    pred_meandif = pred_dif.abs().mean()

    # image results
    plt.figure()
    pred_dif[wr.get_cols(pred_dif,['bidopen'])].abs().mean().plot()
    pred_dif[wr.get_cols(pred_dif,['bidhigh'])].abs().mean().plot()
    pred_dif[wr.get_cols(pred_dif,['bidlow'])].abs().mean().plot()

    # abs error across time
    plt.figure()
    pred_dif[wr.get_cols(pred_dif,['openEUR/USD'])].abs().plot()
    pred_dif.abs().mean(axis=1).plot()

    model.save('ex_nets/predict_markets/pm5_4204')