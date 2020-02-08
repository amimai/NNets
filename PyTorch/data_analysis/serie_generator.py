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
file_data=['EUR_USD','GBP_JPY','AUD_CAD']#['EUR_USD','EUR_GBP','GBP_USD','EUR_JPY','GBP_JPY']


def process_datfile(file_data,spacing):
    print('gathering data')
    # collect data PyTorch/data/finance1m/EUR_USD
    data_dir = 'PyTorch/data/finance1m/'
    for each in file_data:
        # creates spaced data sets
        dat = pd.read_csv(data_dir+each, index_col='date')
        dat = dat[wr.get_cols(dat, ['bidclose','bidhigh','bidlow','tickqty'])]
        dat['bidhigh'] = (dat['bidclose'].rolling(spacing).max() - dat['bidclose'])/dat['bidclose'] # get max
        dat['bidlow'] = (dat['bidclose'].rolling(spacing).min() - dat['bidclose'])/dat['bidclose'] # min
        dat['bidclose'] = (dat['bidclose'].diff().rolling(spacing).sum())/dat['bidclose'] # dv
        dat['tickqty'] = (dat['tickqty'].rolling(spacing).sum()**.1)-1 # ticksum
        dat = dat[spacing:]
        dat.to_csv('PyTorch/data/crunched_f1_f120/'+each+'.csv')

def make_time_code(start,end):
    dat = pd.date_range(start=start, end=end, freq='1min')
    dat = pd.DataFrame(dat, columns=['date'])
    dat['datetime'] = pd.to_datetime(dat['date'])
    dat = dat.set_index('datetime')
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

    dat.drop(['date'], axis=1, inplace=True)
    return dat



def compile_data(file_data,spacing,backwards):
    now = time.time()
    data_dir = 'PyTorch/data/crunched_f1_f120/'
    start = 0
    end = 0

    #build timeseries
    for each in file_data:
        dat = pd.read_csv(data_dir + each+'.csv', index_col='date')
        if not start: start = dat.index[0]
        if not end: end = dat.index[-1]
        if dat.index[0]<start:start = dat.index[0]
        if dat.index[-1]>end:end = dat.index[-1]
    timestamp = make_time_code(start,end)
    tmp = timestamp.iloc[:,1:2].copy() # make placeholder to maintain times

    stds = []
    #build data
    for i in range(len(file_data)):
        dat = pd.read_csv(data_dir + file_data[i] + '.csv' , index_col='date')
        stds.append(dat.std()) #to reverse out the transact
        dat = dat / dat.std()
        tmp = tmp.join(dat, lsuffix='', rsuffix=file_data[i])
    tmp = tmp.iloc[:, 1:] # drop placeholder
    tmp = tmp.fillna(0.0) # zerofill n/a's

    tru = tmp # targets
    dat = timestamp.join(tmp) #data with encoding

    #building labelset
    inputs = np.zeros((len(dat) - backwards * spacing, backwards, dat.shape[-1]))
    labels = np.zeros((len(dat) - backwards * spacing, tru.shape[-1]))
    for i in range(backwards * spacing, len(dat)):
        if i%100000==0: print('building labels ', time.time()-now, ' i = ',i)
        inputs[i - backwards * spacing] = dat[i - backwards * spacing:i][::spacing]
        labels[i - backwards * spacing] = tru[i:i + 1]

    inputs = inputs.reshape(-1, backwards, dat.shape[-1])
    labels = labels.reshape(-1, tru.shape[-1])

    return inputs, labels

def compile_data_ns(file_data,spacing,backwards):
    #lite version using less data
    now = time.time()
    data_dir = 'PyTorch/data/crunched_f1_f120/'
    start = 0
    end = 0
    stds = []
    tmp = pd.read_csv(data_dir + file_data[0] + '.csv' , index_col='date')
    stds.append(tmp.std())  # to reverse out the transact
    tmp = tmp / tmp.std()


    #build data
    for i in range(1,len(file_data)):
        dat = pd.read_csv(data_dir + file_data[i] + '.csv' , index_col='date')
        stds.append(dat.std()) #to reverse out the transact
        dat = dat / dat.std()
        tmp = tmp.join(dat, lsuffix='', rsuffix=file_data[i])
    #tmp = tmp.iloc[:, 1:] # drop placeholder
    tmp = tmp.fillna(0.0) # zerofill n/a's

    dat = tmp
    tru = tmp

    #building labelset
    inputs = np.zeros((len(dat) - backwards * spacing, backwards, dat.shape[-1]))
    labels = np.zeros((len(dat) - backwards * spacing, tru.shape[-1]))
    for i in range(backwards * spacing, len(dat)):
        if i%100000==0: print('building labels ', time.time()-now, ' i = ',i)
        inputs[i - backwards * spacing] = dat[i - backwards * spacing:i][::spacing]
        labels[i - backwards * spacing] = tru[i:i + 1]

    inputs = inputs.reshape(-1, backwards, dat.shape[-1])
    labels = labels.reshape(-1, tru.shape[-1])

    return inputs, labels

if __name__ == '__main__':
    backwards = 128
    spacing = 120  # spacing used to determine the space between each datapoint grouping data between
    file_data = ['EUR_USD', 'GBP_JPY', 'AUD_CAD']  # ['EUR_USD','EUR_GBP','GBP_USD','EUR_JPY','GBP_JPY']
    process_datfile(file_data, spacing)
    inputs, labels = compile_data_ns(file_data,spacing,backwards)
    np.save('PyTorch/data/built_f1_f120/inputs_ns_EUR_USDGBP_JPYAUD_CAD', inputs, allow_pickle=False)
    np.save('PyTorch/data/built_f1_f120/labels_ns_EUR_USDGBP_JPYAUD_CAD', labels, allow_pickle=False)