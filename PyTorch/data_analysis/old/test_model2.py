import os
import time

import pandas as pd
import numpy as np



from PyTorch.data_analysis.build_DataStore import ForexDataset

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size=3000 #300, 17s, .79loss #3000, 3s, .81loss
learn_rate=1e-3
hidden_dim=2**5
EPOCHS=15
n_layers = 2
output_dim = 168


file_data='PyTorch/data/Finance_4b_4f/d_msd.csv'
file_truth='PyTorch/data/Finance_4b_4f/t_msd.csv'

backwards = 48
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


label_scalers = {}

train_x = []
train_y = []
test_x = {}
test_y = {}

# datasplit
test_portion = int(len(inputs)*.1)
if len(train_x) == 0:
    train_x = inputs[:-test_portion]
    train_y = labels[:-test_portion]
else:
    train_x = np.concatenate((train_x, inputs[:-test_portion]))
    train_y = np.concatenate((train_y, labels[:-test_portion]))
test_x = (inputs[-test_portion:])
test_y = (labels[-test_portion:])


train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=8)

test_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=8)

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, backwards, drop_prob=0.2, GPUs=1):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.backwards = backwards
        self.GPUs = GPUs

        self.modlist1 = nn.ModuleList()
        self.modlist1.append( nn.GRU(input_dim, hidden_dim, n_layers,
                                    batch_first=True, dropout=drop_prob) )
        self.modlist1.append( nn.GRU(hidden_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=drop_prob) )
        self.modlist1.append( nn.GRU(hidden_dim, hidden_dim, n_layers,
                                    batch_first=True, dropout=drop_prob) )

        self.modlist2 = nn.ModuleList()
        self.modlist2.append( nn.Linear(hidden_dim*backwards, hidden_dim*backwards) )
        self.modlist2.append( nn.Linear(hidden_dim*backwards, hidden_dim*backwards) )
        self.modlist2.append( nn.Linear(hidden_dim, output_dim) )

        self.modlist3 = nn.ModuleList()
        self.modlist3.append( nn.ReLU() )
        self.modlist3.append( nn.ReLU() )
        self.modlist3.append( nn.ReLU() )

    def forward(self, x, h):
        #print('x0 ',x.shape) #torch.Size([100, 12, 168])
        #print('h0 ',h.shape) #torch.Size([1, 100, 32])

        self.modlist1[0].flatten_parameters()
        out, h = self.modlist1[0](x, h)
        out = out.reshape(-1,self.hidden_dim*self.backwards)
        out = self.modlist2[0](self.modlist3[0](out))
        out = out.reshape(-1,self.backwards,self.hidden_dim)

        self.modlist1[1].flatten_parameters()
        out, h = self.modlist1[1](out, h)
        out = out.reshape(-1, self.hidden_dim * self.backwards)
        out = self.modlist2[1](self.modlist3[1](out))
        out = out.reshape(-1, self.backwards, self.hidden_dim)

        self.modlist1[2].flatten_parameters()
        out, h = self.modlist1[2](out, h)
        #print('out ', out.shape) #torch.Size([100, 12, 32])
        #print('h0 out0 ', h.shape) #torch.Size([1, 100, 32])
        #print('fc1 ', out[:, -1].shape)#torch.Size([100, 32])
        out = self.modlist2[2](self.modlist3[2](out[:, -1]))
        #print('feed ', out.shape) #torch.Size([100, 168])
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(int(self.n_layers * self.GPUs), int(batch_size / self.GPUs), self.hidden_dim).zero_().to(device)
        return hidden

def train(train_loader, learn_rate, test=None, c_model=None, c_hist=None, hidden_dim=256, n_layers = 2, EPOCHS=5):
    # Setting common hyperparameters

    print(len(next(iter(train_loader))))
    print(next(iter(train_loader))[0][0].shape)
    print(next(iter(train_loader))[0][0].shape[1])
    input_dim = next(iter(train_loader))[0][0].shape[1]

    if c_model !=None:
        model = c_model
    else:
        # Instantiating the models
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, backwards, GPUs=3)

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

    history = {'epoch_loss'     :[],
               'epoch_times'    :[],
               'epoch_val'      :[]}

    if c_hist != None:
        history = c_hist

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
        history['epoch_loss'].append(avg_loss / len(train_loader))
        history['epoch_times'].append(current_time - start_time)
        if test != None:
            _o,_t,epoch_val = evaluate(model, test_loader)
            model.train()
            history['epoch_val'].append(epoch_val)
            print('eval loss ', epoch_val)


    print("Total Training Time: {} seconds".format(str(sum(history['epoch_times']))))
    return model, history

def evaluate(model,test_loader, pred = False):
    model.eval()
    with torch.no_grad():
        outputs = []
        targets = []
        h = model.module.init_hidden(batch_size)
        criterion = nn.MSELoss()
        losses = 0
        i=0
        for x, label in test_loader:
            h = h.data
            i+=1
            out, h = model(x.to(device).float(), h)
            if pred:
                outputs.append((out.cpu().detach().numpy()).reshape(-1))
                targets.append((label.numpy()).reshape(-1))
        #print("Evaluation Time: {}".format(str(time.clock() - start_time)))
            loss = criterion(out, label.to(device).float())
            losses+=loss.item()
        loss = losses/i
    return outputs, targets, loss


gru_model, history = train(train_loader, learn_rate, test=test_loader,
                  hidden_dim=hidden_dim, n_layers = n_layers,
                  EPOCHS=EPOCHS)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(history['epoch_loss'])
    plt.plot(history['epoch_times'])
    plt.plot(history['epoch_val'])