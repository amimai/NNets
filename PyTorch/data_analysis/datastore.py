import os
import time
import math

import pandas as pd
import numpy as np
from DataProcess import wrangle as wr


#from PyTorch.data_analysis.build_DataStore import ForexDataset

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from datetime import datetime



now = time.time()
backwards = 128
spacing = 120 #spacing used to determine the space between each datapoint grouping data between
if not True: # for making a new datafile

    print('gathering data')
    # collect data PyTorch/data/finance1m/EUR_USD
    data_dir = 'PyTorch/data/finance1m/'
    file_data=['EUR_USD','GBP_JPY','AUD_CAD']#['EUR_USD','EUR_GBP','GBP_USD','EUR_JPY','GBP_JPY']
    dat = pd.read_csv(data_dir+file_data[0], index_col='date')
    dat = dat.join(pd.read_csv(data_dir+file_data[1], index_col='date'),lsuffix=file_data[0],rsuffix=file_data[1])
    for i in range(1,len(file_data)):
        dat = dat.join(pd.read_csv(data_dir+file_data[i], index_col='date'),lsuffix='',rsuffix=file_data[i])

    print('dropping data', time.time()-now)
    # drop unnesesary columns
    good_cols = wr.get_cols(dat, ['bidclose','bidhigh','bidlow'])#,'tick'])#['EUR/USD'])  # ['bidopen'])# , 'tick'
    dat = dat[good_cols]

    print('cleaning data', time.time()-now)
    # clean up our data and fill the gaps that are left (from market shutdown over weekend)
    dat = dat.fillna(method='ffill')
    dat = dat.fillna(method='bfill')

    print('manipulating data', time.time()-now)
    # diffrence and manipulate
    dat = (dat.diff()/dat)[1:]
    dat = dat.rolling(spacing).sum()[spacing:]
    dat = dat/dat.std()
    tru = dat.copy()

    print('building timestamp data', time.time()-now)
    # encode timestamp
    dat['p_e_m1'] =  pd.to_datetime(dat.index)
    dat['p_e_m1'] =  dat['p_e_m1'].apply(lambda x:np.sin((np.pi*2/12)* x.date().month))
    dat['p_e_d1'] =  pd.to_datetime(dat.index)
    dat['p_e_d1'] =  dat['p_e_d1'].apply(lambda x:np.sin((np.pi*2/31)* x.date().weekday()))
    dat['p_e_H1'] =  pd.to_datetime(dat.index)
    dat['p_e_H1'] =  dat['p_e_H1'].apply(lambda x:np.sin((np.pi*2/24)* x.hour))
    dat['p_e_M1'] =  pd.to_datetime(dat.index)
    dat['p_e_M1'] =  dat['p_e_M1'].apply(lambda x:np.sin((np.pi*2/7)* x.minute))

    dat['p_e_m2'] =  pd.to_datetime(dat.index)
    dat['p_e_m2'] =  dat['p_e_m2'].apply(lambda x:np.cos((np.pi*2/12)* x.date().month))
    dat['p_e_d2'] =  pd.to_datetime(dat.index)
    dat['p_e_d2'] =  dat['p_e_d2'].apply(lambda x:np.cos((np.pi*2/31)* x.date().weekday()))
    dat['p_e_H2'] =  pd.to_datetime(dat.index)
    dat['p_e_H2'] =  dat['p_e_H2'].apply(lambda x:np.cos((np.pi*2/24)* x.hour))
    dat['p_e_M2'] =  pd.to_datetime(dat.index)
    dat['p_e_M2'] =  dat['p_e_M2'].apply(lambda x:np.cos((np.pi*2/7)* x.minute))



    #building labelset
    inputs = np.zeros((len(dat) - backwards * spacing, backwards, dat.shape[-1]))
    labels = np.zeros((len(dat) - backwards * spacing, tru.shape[-1]))
    for i in range(backwards * spacing, len(dat)):
        if i%100000==0: print('building labels ', time.time()-now, ' i = ',i)
        inputs[i - backwards * spacing] = dat[i - backwards * spacing:i][::spacing]
        labels[i - backwards * spacing] = tru[i:i + 1]

    inputs = inputs.reshape(-1, backwards, dat.shape[-1])
    labels = labels.reshape(-1, tru.shape[-1])

    del dat
    del tru
    np.save('PyTorch/data/built_finance/inputs_EUR_USDGBP_JPYAUD_CAD', inputs, allow_pickle=False)
    np.save('PyTorch/data/built_finance/labels_EUR_USDGBP_JPYAUD_CAD', labels, allow_pickle=False)
else:
    inputs = np.load('PyTorch/data/built_finance/inputs_EUR_USDGBP_JPYAUD_CAD.npy')
    labels = np.load('PyTorch/data/built_finance/labels_EUR_USDGBP_JPYAUD_CAD.npy')

labels = labels.clip(min=-2.,max=2.)
labels = np.delete(labels, np.arange(3,labels.size,4)).reshape(*labels.shape[:-1],-1) #remove ticks from targets
#labels = labels[:,::3] # only look at opens

train_x = []
train_y = []
test_x = {}
test_y = {}

print('splitting test labels', time.time()-now)
# datasplit
test_portion = int(len(inputs)*.01)
if len(train_x) == 0:
    train_x = inputs[:-test_portion]
    train_y = labels[:-test_portion]
else:
    train_x = np.concatenate((train_x, inputs[:-test_portion]))
    train_y = np.concatenate((train_y, labels[:-test_portion]))
test_x = (inputs[-test_portion:])
test_y = (labels[-test_portion:])
del inputs
del labels

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
del train_x
del train_y

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
del test_x
del test_y