import pandas as pd

from DataProcess import wrangle as wr
from DataProcess import datasets as ds

from Networks import lstm_stateful as l


data_raw = pd.read_csv('fxcm_all_10k.csv',index_col='date')
data = wr.mean_normalize(wr.differance(data_raw))

dataset = ds.to_dataset(data)

# offset by 1
train, test, val = ds.seq_stateful(dataset[:-1],dataset[1:],0.1,0.1)

model = l.model_lstm_seq(train[0][:-1],train[0][1:],500,1)
model2 = l.model_lstm_seq_accord(train[0][:-1],train[0][1:],400,2)


def run(epochs,train,val):
    l.train(model, train[0], train[1], val[0], val[1], epochs, 8)

def run2(epochs,train,val):
    l.train(model2, train[0], train[1], val[0], val[1], epochs, 8)