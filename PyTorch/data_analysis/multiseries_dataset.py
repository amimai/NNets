import time
import datetime
import fxcmpy
from DataProcess import wrangle as wr

import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

class datacompiler:
    def __init__(self):
        self.instruments = ['EUR/USD']
        self.colums = ['bidclose', 'bidhigh', 'bidlow', 'tickqty']
        self.spacings = [5,30,240]
        self.backwards = [30,30,30]

        self.datafile = []
        data_dir = 'PyTorch/data/finance1m/'
        for i in range(len(self.instruments)):
            self.datafile.append(pd.read_csv(data_dir + self.instruments[i].replace('/','_'), index_col='date'))
            self.datafile[i] = self.datafile[i][wr.get_cols( self.datafile[i], ['bidclose', 'bidhigh', 'bidlow', 'tickqty'])]

    def difdat(self,spacing):
        ret = []
        for i in range(len(self.instruments)):
            dat = self.datafile[i].copy()

            dat['bidhigh'] = (dat['bidclose'].rolling(spacing).max() - dat['bidclose']) / dat[
                'bidclose']  # get max
            dat['bidlow'] = (dat['bidclose'].rolling(spacing).min() - dat['bidclose']) / dat['bidclose']  # min
            dat['bidclose'] = (dat['bidclose'].diff().rolling(spacing).sum()) / dat['bidclose']  # dv
            dat['tickqty'] = (dat['tickqty'].rolling(spacing).sum() ** .1) - 1  # ticksum
            dat = dat[max(self.spacings):]
            ret.append(dat)
        return ret

    def norm_dat(self,spacing):
        stds = []
        dat = self.difdat(spacing)
        for i in range(len(dat)):
            stds.append(dat[i].std())
        return stds #returns list of stds

    def stack_timeseries(self):
        ret = []
        for i in range(len(self.spacings)):
            dat = self.difdat(self.spacings[i])
            stds = self.norm_dat(self.spacings[i])
            for i in range(len(dat)):
                dat[i] = dat[i]/stds[i]
            ret.append(dat)
        return ret

    def reverse_pred(self,pred):
        ret = []
        stds = [self.norm_dat(n) for n in self.spacings]
        for i in range(len(pred)):
            pdcopy = pred[i].copy()
            pdcopy.columns = self.datafile[0].columns[:3]
            pdcopy = (pdcopy)*stds[i][0][:3]#/pdcopy.std()
            ret.append(pdcopy)
        return ret

    def create_dataset(self):
        inputs_r = []
        labels_r = []
        stack = self.stack_timeseries()
        for i in range(len(self.spacings)):
            spacing = self.spacings[i]
            backward = self.backwards[i]
            dat = stack[i][0].copy()
            for i in range(spacing): #add empty data to start to make a
                dat = dat.append(pd.Series(), ignore_index=True)
            dat = dat.fillna(0.0)
            tru = dat
            # building labelset
            inputs = np.zeros((len(dat) - backward * spacing, backward, dat.shape[-1]))
            labels = np.zeros((len(dat) - backward * spacing, tru.shape[-1]))
            for i in range(backward * spacing, len(dat)):
                inputs[i - backward * spacing] = dat[i - backward * spacing:i][::spacing]
                labels[i - backward * spacing] = tru[i:i + 1]

            inputs = inputs.reshape(-1, backward, dat.shape[-1])
            labels = labels.reshape(-1, tru.shape[-1])
            inputs_r.append(inputs)
            labels_r.append(labels)
        return inputs_r,labels_r


if __name__ == '__main__':
    from PyTorch.data_analysis.multiseries_dataset import *
    d = datacompiler()
    x,y = d.create_dataset()
    np.save('PyTorch/data/built_finance/eurostack/inputs0', x[0], allow_pickle=False)
    np.save('PyTorch/data/built_finance/eurostack/inputs1', x[1], allow_pickle=False)
    np.save('PyTorch/data/built_finance/eurostack/inputs2', x[2], allow_pickle=False)
    np.save('PyTorch/data/built_finance/eurostack/labels0', y[0], allow_pickle=False)
    np.save('PyTorch/data/built_finance/eurostack/labels1', y[1], allow_pickle=False)
    np.save('PyTorch/data/built_finance/eurostack/labels2', y[2], allow_pickle=False)

    from PyTorch.data_analysis.multiseries_dataset import *
    from PyTorch.data_analysis.test_model10e import *
    savedir = 'PyTorch/data_analysis/saved_mod/test_10/e'

    y2 = np.load('PyTorch/data/built_finance/eurostack/labels2.npy').clip(min=-2.,max=2.)[:,:3]
    y0 = np.load('PyTorch/data/built_finance/eurostack/labels0.npy')[-len(y2):].clip(min=-2.,max=2.)[:,:3]
    y1 = np.load('PyTorch/data/built_finance/eurostack/labels1.npy')[-len(y2):].clip(min=-2.,max=2.)[:,:3]
    x0 = np.load('PyTorch/data/built_finance/eurostack/inputs0.npy')[-len(y2):].clip(min=-2.,max=2.)
    x1 = np.load('PyTorch/data/built_finance/eurostack/inputs1.npy')[-len(y2):].clip(min=-2.,max=2.)
    x2 = np.load('PyTorch/data/built_finance/eurostack/inputs2.npy')[-len(y2):].clip(min=-2.,max=2.)

    test_portion = int(len(y0) * .01)
    dataset = TensorDataset(torch.from_numpy(x0)[:-test_portion], torch.from_numpy(x1)[:-test_portion], torch.from_numpy(x2)[:-test_portion],
                            torch.from_numpy(y0)[:-test_portion], torch.from_numpy(y1)[:-test_portion], torch.from_numpy(y2)[:-test_portion])
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=16)


    test_dataset = TensorDataset(torch.from_numpy(x0)[-test_portion:], torch.from_numpy(x1)[-test_portion:], torch.from_numpy(x2)[-test_portion:],
                            torch.from_numpy(y0)[-test_portion:], torch.from_numpy(y1)[-test_portion:], torch.from_numpy(y2)[-test_portion:])
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=16)

    train = trainer()
    train.boot(batch_size, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards,
               GPUs=[0], trainloader=dataloader)
    train.load(savedir,3)
    for i in range(9):
        train.train(batch_size, EPOCHS=3, test_loader=test_dataloader)
        train.save(savedir)

        pred, targ, loss = train.evaluate(test_dataloader, 0, pred=True)
        df_p0 = np.array(pred)[:, 0, :, :].reshape(-1, 3)
        df_p1 = np.array(pred)[:, 1, :, :].reshape(-1, 3)
        df_p2 = np.array(pred)[:, 2, :, :].reshape(-1, 3)
        df_t0 = np.array(targ)[:, 0, :, :].reshape(-1, 3)
        df_t1 = np.array(targ)[:, 1, :, :].reshape(-1, 3)
        df_t2 = np.array(targ)[:, 2, :, :].reshape(-1, 3)

        dt_arr = [(df_p0, df_t0, 0.8), (df_p1, df_t1, 0.8), (df_p2, df_t2, .2)]
        dt_ret = []
        for dt in dt_arr:
            t = 14000
            datp = pd.DataFrame(dt[0])
            datt = pd.DataFrame(dt[1])
            datp /= datp.std()
            datt /= datt.std()

            thresh = 0.1
            be = dt[2]
            tmp_arr = []
            for n in range(10):
                bl1 = datp.iloc[:t, 0] > thresh
                bl2 = datp.iloc[:t, 0] < (-1 * thresh)
                tot1 = bl1 * (datt.iloc[:t, 0] - be)
                tot2 = bl2 * -1 * (datt.iloc[:t, 0] + be)
                tot3 = tot1 + tot2
                tmp_arr.append(tot3.sum())
                thresh = thresh * 2
            dt_ret.append(max(tmp_arr))
        print (dt_ret)


    train.graph()
    pd.DataFrame(np.array(train.raw_dump)).plot()


    psdataset = datacompiler()
    rev2 = psdataset.reverse_pred([pd.DataFrame(df_t0), pd.DataFrame(df_t1), pd.DataFrame(df_t2)])


    train.load(savedir, 36)
    pred, targ, loss = train.evaluate(test_dataloader, 0, pred=True)
    df_p0 = np.array(pred)[:,0,:,:].reshape(-1,3)
    df_p1 = np.array(pred)[:, 1, :, :].reshape(-1, 3)
    df_p2 = np.array(pred)[:, 2, :, :].reshape(-1, 3)
    df_t0 = np.array(targ)[:, 0, :, :].reshape(-1, 3)
    df_t1= np.array(targ)[:, 1, :, :].reshape(-1, 3)
    df_t2 = np.array(targ)[:, 2, :, :].reshape(-1, 3)

    from matplotlib import pyplot as plt
    t=14000
    datp=pd.DataFrame(df_p2)
    datt=pd.DataFrame(df_t2)
    datp/=datp.std()
    datt/=datt.std()

    thresh = 0.1
    fig4 = plt.figure()
    be=.15
    for i in range(10):
        bl1 = datp.iloc[:t, 0] > thresh
        bl2 = datp.iloc[:t, 0] < (-1 * thresh)
        tot1 = bl1 * (datt.iloc[:t, 0] -be)
        tot2 = bl2 * -1 * (datt.iloc[:t, 0] +be)
        tot3 = tot1 + tot2
        plt.plot(tot3.cumsum())
        thresh = thresh * 2





    fig1 = plt.figure()
    rs_p = datt.iloc[:t,0].rolling(10).sum()
    rs_t = datp.iloc[:t,0].rolling(10).sum()
    plt.plot(rs_p)
    plt.plot(rs_t)

    fig2 = plt.figure()
    plt.plot(datt.iloc[:t,0].cumsum())
    plt.plot(datp.iloc[:t,0].cumsum())
    thresh=0.5
    fig3 = plt.figure()
    plt.plot(((datp.iloc[:t, 0]>thresh)*datt.iloc[:t, 0].abs() + (datp.iloc[:t, 0]<-thresh)*datt.iloc[:t, 0].abs()).cumsum())
    for i in range(4):
        plt.plot(((datp.iloc[:t, 0]>thresh)*datt.iloc[:t, 0] + (datp.iloc[:t, 0]<-thresh)*datt.iloc[:t, 0]).cumsum())
        thresh*=2

    pass

#ep33:.1-.6 yeild return
#ep36:.1-1.2 yeild
#ep39: yeild drops
#42 : remains stable
#45 54 : fails
#57 low yeild

#w L3_MSE_2
#39 still stable but again yeild drops
#42 still stable
#60 is another positive model

#87 w mse +1400

'''
p2
ep@42&60 be@0.2 @thresh 0.16 && only stable @p2 be below 0.08
ep87 be@.2 @thresh 0.8-0.16 && marginal reliable on p0 and p1 @be0.08
p1
'''
#93 @.5 be @thresh .8-1.6

#mselong
# look at ep 15 and ep 60-63 & 90-96, 108

'''
10b
pure mse [mse_long] ep108
l3_mstd_msea + mse ep 87 - slight less stable but higher delta gain
'''

'''
10d-mse2e6
model 12 gained +400

'''