import pandas as pd

from DataProcess import wrangle as wr
from DataProcess import datasets as ds

from Networks import deep_autoencoder as au
from Networks import deeply_connected as dc
from keras.layers import Dense



data_raw = pd.read_csv('fxcm_all_10k.csv',index_col='date')
data = wr.mean_normalize(wr.differance(data_raw))

dataset = ds.to_dataset(data)

train, test, val = ds.random_TTV(dataset,dataset,0.1,0.1)


autoencoder = au.model_autoencoder(train[0],train[1],50,
                                   deep_stacks=[[dc.deepcon_dense, dc.core, 3, 80],
                                                [dc.deepcon_dense, dc.core, 4, 70],
                                                [dc.deepcon_dense, dc.core, 5, 60]],
                                   pool_stacks=[[Dense, 600, 'relu'],
                                                [Dense, 300, 'relu'],
                                                [Dense, 100, 'relu']])

def run(epochs,train,val):
    au.train(autoencoder[0],train[0],train[1],val[0],val[1],epochs,2,7)


