import os
import time

import pandas as pd
import numpy as np



from PyTorch.data_analysis.build_DataStore import ForexDataset

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size=300
learn_rate=1e-3
hidden_dim=2**5
EPOCHS=50
n_layers = 1
output_dim = 168


file_data='PyTorch/data/Finance_4b_4f/d_msd.csv'
file_truth='PyTorch/data/Finance_4b_4f/t_msd.csv'

backwards = 12
forward = 4

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


train_data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=8)


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2, GPUs=1):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.GPUs = GPUs

        self.gru1 = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                          dropout=drop_prob)
        self.gru2 = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                          dropout=drop_prob)
        self.gru3 = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                          dropout=drop_prob)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        #print('x0 ',x.shape) #torch.Size([100, 12, 168])
        #print('h0 ',h.shape) #torch.Size([1, 100, 32])
        out, h = self.gru1(x, h)
        #print('out ', out.shape) #torch.Size([100, 12, 32])
        #print('h0 out0 ', h.shape) #torch.Size([1, 100, 32])
        #print('fc1 ', out[:, -1].shape)#torch.Size([100, 32])
        out = self.fc3(self.relu(out[:, -1]))
        #print('feed ', out.shape) #torch.Size([100, 168])
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(int(self.n_layers * self.GPUs), int(batch_size / self.GPUs), self.hidden_dim).zero_().to(device)
        return hidden

def train(train_loader, learn_rate, hidden_dim=256, n_layers = 2, EPOCHS=5):
    # Setting common hyperparameters

    print(len(next(iter(train_loader))))
    print(next(iter(train_loader))[0][0].shape)
    print(next(iter(train_loader))[0][0].shape[1])
    input_dim = next(iter(train_loader))[0][0].shape[1]


    # Instantiating the models
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, GPUs=3)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=[0, 0, 1])
        # max batdch size config = [0,0,1] [bs=11000] , max speed config = 0,0,0,1 bs = [3000]

    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format('''GRU'''))
    epoch_times = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.clock()
        h = model.module.init_hidden(batch_size)  # add .module for dataparallele to reach model original inside
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1

            h = h.data

            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model

gru_model = train(train_loader, learn_rate,
                  hidden_dim=hidden_dim, n_layers = n_layers,
                  EPOCHS=EPOCHS)