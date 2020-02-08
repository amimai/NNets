import time
import datetime
import fxcmpy
from DataProcess import wrangle as wr

import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader



class dataholder:
    def __init__(self):
        self.TOKEN = '85902b8e33d2198b8a99769de6e820b85a42e4b7'
        self.colums = ['bidclose', 'bidhigh', 'bidlow', 'tickqty']
        self.instruments = ['EUR/USD','GBP/JPY','AUD/CAD']

        self.spacing = 120
        self.backwards = 128

        self.datafile = []
        self.con = fxcmpy.fxcmpy(access_token=self.TOKEN, log_level='error')

        for i in range(len(self.instruments)):
            self.datafile.append(self.con.get_candles(self.instruments[i], columns=self.colums, period='m1', number=9000))
        for _ in range(3):
            for i in range(len(self.instruments)):
                self.datafile[i] =  (self.con.get_candles(self.instruments[1], columns=self.colums, period='m1',
                                     start=self.datafile[i].index.min()- datetime.timedelta(6), end=self.datafile[i].index.min())).append(self.datafile[i]).drop_duplicates()

        self.stds = self.get_normaliser()

    def get_normaliser(self):
        file_data = [i.replace('/','_') for i in self.instruments ]
        data_dir = 'PyTorch/data/crunched_f1_f120/'
        stds = []

        tmp = pd.read_csv(data_dir + file_data[0] + '.csv', index_col='date')
        stds.append(tmp.std())  # to reverse out the transact

        # build data
        for i in range(1, len(file_data)):
            dat = pd.read_csv(data_dir + file_data[i] + '.csv', index_col='date')
            stds.append(dat.std())  # to reverse out the transact
        return stds

    def extend(self):
        for i in range(len(self.instruments)):
            self.datafile[i] = self.datafile[i].append(
                self.con.get_candles(self.instruments[i], columns=self.colums, period='m1', start=self.datafile[i].index.max(),stop=datetime.datetime.now())
            ).drop_duplicates()

    def return_difdat(self):
        ret = []
        for each in self.datafile:
            # creates spaced data sets
            dat = each.copy()
            dat['bidhigh'] = (dat['bidclose'].rolling(self.spacing).max() - dat['bidclose']) / dat['bidclose']  # get max
            dat['bidlow'] = (dat['bidclose'].rolling(self.spacing).min() - dat['bidclose']) / dat['bidclose']  # min
            dat['bidclose'] = (dat['bidclose'].diff().rolling(self.spacing).sum()) / dat['bidclose']  # dv
            dat['tickqty'] = (dat['tickqty'].rolling(self.spacing).sum() ** .1) - 1  # ticksum
            dat = dat[self.spacing:]
            ret.append(dat)
        return ret

    def buid_dataset(self):
        data = self.return_difdat()
        tmp = data[0]/self.stds[0]
        for i in range(1,len(data)):
            dat = data[i]/self.stds[i]
            tmp = tmp.join(dat, lsuffix='', rsuffix=self.instruments[i].replace('/','_'))
        tmp = tmp.fillna(0.0)  # zerofill n/a's
        for _ in range(self.spacing): #stack empty frames to move time
            tmp = tmp.append(pd.Series(), ignore_index=True)

        dat = tmp
        tru = tmp

        # building labelset
        inputs = np.zeros((len(dat) - self.backwards * self.spacing, self.backwards, dat.shape[-1]))
        labels = np.zeros((len(dat) - self.backwards * self.spacing, tru.shape[-1]))
        for i in range(self.backwards * self.spacing, len(dat)):
            inputs[i - self.backwards * self.spacing] = dat[i - self.backwards * self.spacing:i][::self.spacing]
            labels[i - self.backwards * self.spacing] = tru[i:i + 1]

        inputs = inputs.reshape(-1, self.backwards, dat.shape[-1])
        labels = labels.reshape(-1, tru.shape[-1])
        return inputs,labels

    def get_dataloaders(self):
        inputs,labels = self.buid_dataset()
        labels = np.delete(labels, np.arange(3, labels.size, 4)).reshape(*labels.shape[:-1], -1)
        dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
        dataloaders = DataLoader(dataset, shuffle=False, batch_size=1, drop_last=True, num_workers=16)
        return dataloaders

if __name__ == '__main__':
    from PyTorch.data_analysis.test_model9c import trainer
    from PyTorch.data_analysis.datafetch import dataholder
    dl = dataholder()
    dlers = dl.get_dataloaders()
    train = trainer()
    train.boot(1, hidden_dim=2 ** 6, output_dim=9, n_layers=4, backwards=128,GPUs = [0], trainloader = dlers)
    savedir = 'PyTorch/data_analysis/saved_mod/test_9/c'
    train.load(savedir, 33)
    pred, targ, loss = train.evaluate(dlers, 0, pred=True)

    del pred, targ, loss, dlers
    dl.extend()
    dlers = dl.get_dataloaders()
    pred, targ, loss = train.evaluate(dlers, 0, pred=True)
    df_p = pd.DataFrame(pred)
    df_t = pd.DataFrame(targ).fillna(0.0)
    df_d = df_p - df_t

    t = 0
    up = (((df_p.iloc[:, t] > .2)) * df_t.iloc[:, t])
    down = (((df_p.iloc[:, t] < -.2)) * -1 * df_t.iloc[:, t])
    (up + down).cumsum().plot()
    up = (((df_p.iloc[:, t].rolling(15).sum() > .2)) * df_t.iloc[:, t])
    down = (((df_p.iloc[:, t].rolling(15).sum() < -.2)) * -1 * df_t.iloc[:, t])
    (up + down).cumsum().plot()
    up = (((((df_p.iloc[:, t + 1] ** 2) < (df_p.iloc[:, t + 2] ** 6)))) * df_t.iloc[:, t])
    down = (((((df_p.iloc[:, t + 1] ** 2) > (df_p.iloc[:, t + 2] ** 6)))) * -1 * df_t.iloc[:, t])
    (up + down).cumsum().plot()
    up = (((((df_p.iloc[:, t + 1] ** 2) < (df_p.iloc[:, t + 2] ** 2)))) * df_t.iloc[:, t])
    down = (((((df_p.iloc[:, t + 1] ** 2) > (df_p.iloc[:, t + 2] ** 2)))) * -1 * df_t.iloc[:, t])
    (up + down).cumsum().plot()
    up = (((((df_p.iloc[:, t + 1] ** 2) + .25 < (df_p.iloc[:, t + 2] ** 2)))) * df_t.iloc[:, t])
    down = (((((df_p.iloc[:, t + 1] ** 2) + .25 > (df_p.iloc[:, t + 2] ** 2)))) * -1 * df_t.iloc[:, t])
    (up + down).cumsum().plot()

    t=0
    b = 240
    df_p.iloc[-b:, t].plot()
    (df_p.iloc[-b:, t+1]+df_p.iloc[-b:, t]).plot()
    (df_p.iloc[-b:, t+2]+df_p.iloc[-b:, t]).plot()

    t=0
    b = 480
    (df_p.iloc[-b:, t+2]**2).plot()
    (df_p.iloc[-b:, t+1]**2).plot()
    df_t.iloc[-b:, t].plot()

    t=3
    b = 60
    p = df_p.iloc[-b:, :]
    pup = ((p.iloc[:, t + 1] ** 2) + .25 < (p.iloc[:, t + 2] ** 2)) #(p.iloc[:, t].rolling(15).sum() > .2)#((p.iloc[:, t + 1] ** 2) < (p.iloc[:, t + 2] ** 6))
    #(df_p.iloc[:, t] > .2)  #((p.iloc[:, t + 1] ** 2) < (p.iloc[:, t + 2] ** 2)) #
    pdo = ((p.iloc[:, t + 1] ** 2) + .25 > (p.iloc[:, t + 2] ** 2)) #(p.iloc[:, t].rolling(15).sum() < -.2)#((p.iloc[:, t + 1] ** 2) > (p.iloc[:, t + 2] ** 6))
    #(df_p.iloc[:, t] < -.2) #((p.iloc[:, t + 1] ** 2) > (p.iloc[:, t + 2] ** 2)) #