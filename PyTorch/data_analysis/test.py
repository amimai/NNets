import pandas as pd
import numpy as np

file_data='PyTorch/data/Finance_4b_4f/d_msd.csv'
file_truth='PyTorch/data/Finance_4b_4f/t_msd.csv'

backwards = 12
forward = 12

dat = pd.read_csv(file_data, index_col='date')
tru = pd.read_csv(file_truth, index_col='date')

lookback = backwards
inputs = np.zeros((len(dat) - lookback, lookback, dat.shape[-1]))
labels = np.zeros((len(dat) - lookback,tru.shape[-1]))

for i in range(lookback, len(dat)-forward):
    inputs[i - lookback] = dat[i - lookback:i]
    labels[i - lookback] = tru[i+forward-1:i+forward]
inputs = inputs.reshape(-1, lookback, dat.shape[-1])
labels = labels.reshape(-1, tru.shape[-1])

batch_size = 1200
train_data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=8)
