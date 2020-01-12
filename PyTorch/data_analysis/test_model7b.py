import os
import time

import pandas as pd
import numpy as np
from DataProcess import wrangle as wr


#from PyTorch.data_analysis.build_DataStore import ForexDataset

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size=3**4 #300, 17s, .79loss #3000, 3s, .81loss
learn_rate=1e-2
hidden_dim=2**9
EPOCHS=100
n_layers = 3
output_dim = 168


file_data='PyTorch/data/linear/d_msd.csv'
file_truth='PyTorch/data/linear/t_msd.csv'

backwards = 12
forward = 3

# collect data
file_data='all_data_223k_3y_m5.csv'
dat = pd.read_csv(file_data, index_col='date')

bad_instruments = ['FRA40', 'CHN50', 'US2000', 'USOil', 'SOYF', 'WHEATF', 'CORNF', 'EMBasket', 'JPYBasket',
                       'BTC/USD', 'BCH/USD', 'ETH/USD', 'LTC/USD', 'XRP/USD', 'CryptoMajor', 'USEquities']
bad_cols = wr.get_cols(dat, bad_instruments)

# clean up our data and fill the gaps that are left (from market shutdown over weekend)
dat = dat.drop(bad_cols, axis=1)
dat = dat.fillna(method='ffill')
dat = dat.fillna(method='bfill')

good_cols = wr.get_cols(dat, ['bidopen','bidhigh','bidlow'])#['EUR/USD'])  # ['bidopen'])# , 'tick'
dat = dat[good_cols]
dat = dat.diff()[1:]
tru = dat.rolling(forward).sum()
dat,tru = dat/dat.std(), tru/tru.std()

#d_mean = dat.mean()
#d_std = dat.std()
#dat = ((dat-d_mean)/d_std)+(d_mean/d_std)


#tru = pd.read_csv(file_truth, index_col='date')



lookback = backwards
inputs = np.zeros((len(dat) - lookback, lookback, dat.shape[-1]))
labels = np.zeros((len(dat) - lookback, tru.shape[-1]))

for i in range(lookback, len(dat)-forward):
    inputs[i - lookback] = dat[i - lookback:i]
    labels[i - lookback] = tru[i-1+forward:i+forward]
    #labels[i - lookback] = tru[i - 1 :i ] # for testing
inputs = inputs.reshape(-1, lookback, dat.shape[-1])
labels = labels.reshape(-1, tru.shape[-1])

dat = None
tru = None

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
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=16)

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=16)

from torch.nn import functional as F

def transform_loader():
    n_x = train_x * (((np.random.rand(*train_x.shape)-.5)/10)+1)
    n_y = train_y * (((np.random.rand(*train_y.shape)-.5)/10)+1)

    n_train_data = TensorDataset(torch.from_numpy(n_x), torch.from_numpy(n_y))
    n_x = None
    n_y = None
    n_train_loader = DataLoader(n_train_data, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=16)
    return n_train_loader






class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, backwards, drop_prob=0.2, GPUs=1):
        super(GRUNet, self).__init__()
        assert backwards%4==0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.backwards = backwards
        self.GPUs = GPUs
        self.drop_prob = drop_prob

        self.relus = nn.ModuleList()
        self.drops = nn.ModuleList()


        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()

        self.c1_lin1 = nn.ModuleList()
        self.c2_lin1 = nn.ModuleList()
        self.c1_lin2 = nn.ModuleList()
        self.c2_lin2 = nn.ModuleList()
        self.c_out = nn.ModuleList()

        #shallow convolved stack
        for i in range(self.n_layers):
            z0_dim = 2 ** i
            z1_dim = 2 ** (i + 1)
            if i == 0 :
                self.conv1.append(nn.Conv2d(z0_dim, z1_dim, (2, 3), stride=(1, 3)))  # concat the open/high/low into one value
            else:
                self.conv1.append(nn.Conv2d(z0_dim, z1_dim, (2, 1), stride=(1, 1)))
                #d = z1_dim*56*(backwards-1*(i+1))
            self.conv2.append(nn.Conv2d(z0_dim, z1_dim, (3, 1), stride=(1, 1))) # convolve data lengthwise
            #d = z1_dim*168*(backwards-2*(i+1))

            for _ in range(2):
                self.relus.append(nn.ReLU())
                self.drops.append(nn.Dropout(p=self.drop_prob))
            for _ in range(5):
                self.relus.append(nn.ReLU())

            self.c1_lin1.append(nn.Linear(z1_dim*56*(backwards-1*(i+1)),hidden_dim))
            self.c1_lin2.append(nn.Linear(hidden_dim, hidden_dim))
            self.c2_lin1.append(nn.Linear(z1_dim*168*(backwards-2*(i+1)),hidden_dim))
            self.c2_lin2.append(nn.Linear(hidden_dim, hidden_dim))
            self.c_out.append(nn.Linear(hidden_dim,hidden_dim))




        self.out_add = nn.Linear(backwards*input_dim,self.n_layers* hidden_dim)
        self.out_pre = nn.Linear(self.n_layers*hidden_dim,self.n_layers* hidden_dim)
        self.out_lin = nn.Linear(self.n_layers* hidden_dim,output_dim)

    def forward(self, x):
        drops = 0
        rels = 0

        for i in range(self.n_layers):
            z1_dim = 2 ** (i + 1)
            xi = x.view(-1, 1, self.backwards, self.input_dim)
            # do convolutions
            if i == 0:
                c1 = self.drops[drops](self.relus[rels](self.conv1[i](xi)))
                c2 = self.drops[drops+1](self.relus[rels+1](self.conv2[i](xi)))
                c1l = self.relus[rels + 2](self.c1_lin1[i](c1.view(-1, z1_dim * 56 * (self.backwards - 1 * (i + 1)))))
                c1l = self.relus[rels + 3](self.c1_lin2[i](c1l))
                c2l = self.relus[rels + 4](self.c2_lin1[i](c2.view(-1, z1_dim * 168 * (self.backwards - 2 * (i + 1)))))
                c2l = self.relus[rels + 5](self.c2_lin2[i](c2l))
                cat = self.relus[rels + 6](self.c_out[i](torch.add(c1l, c2l)))
            else:
                c1 = self.drops[i](self.relus[i](self.conv1[i](c1)))
                c2 = self.drops[i+1](self.relus[i+1](self.conv2[i](c2)))
                c1l = self.relus[rels + 2](self.c1_lin1[i](c1.view(-1, z1_dim * 56 * (self.backwards - 1 * (i + 1)))))
                c1l = self.relus[rels + 3](self.c1_lin2[i](c1l))
                c2l = self.relus[rels + 4](self.c2_lin1[i](c2.view(-1, z1_dim * 168 * (self.backwards - 2 * (i + 1)))))
                c2l = self.relus[rels + 5](self.c2_lin2[i](c2l))
                cat = torch.cat((self.relus[rels + 6](self.c_out[i](torch.add(c1l, c2l))),cat),1)

            drops += 2
            rels += 7

        out = self.out_add(x.view(-1,self.backwards*self.input_dim))
        out = self.out_pre(torch.add(cat,out))
        out = self.out_lin(F.relu(out))

        return out

class trainer:
    def __init__(self, model=None, history=None, optimizer = None, scheduler=None, lr=1e-3):
        self.model = model
        self.history = history
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.epoch = 0
        self.scaling = .85
        self.aneal_rate = 3**3
        self.coshot_cycle = 1000


    def drop(self):
        self.model=None
        self.optimizer=None
        self.history=None
        self.epoch=0

    def boot(self,train_loader,hidden_dim=256,output_dim=168,n_layers=2,backwards=48, GPUs=[0]):
        if self.model is None:
            input_dim = next(iter(train_loader))[0][0].shape[1]
            self.model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, backwards, GPUs=len(GPUs))
        if self.history is None:
            self.history = {'epoch_loss':   [],
                            'epoch_times':  [],
                            'epoch_val':    [],
                            'epoch_lr':     []}
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,momentum=0.9,nesterov=True)
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.coshot_cycle, T_mult=1, eta_min=0, last_epoch=-1)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model, device_ids=GPUs)
        self.model.to(device)
        summary(self.model, (120, backwards, input_dim))

    def moving_average(self,data_set, periods=3):
        weights = np.ones(periods) / periods
        return np.convolve(data_set, weights, mode='valid')

    def aneal(self):
        np_arr_d3 = self.moving_average(np.diff(np.array(self.history['epoch_loss'][-self.aneal_rate:]), 1),periods=3).argsort()[:3] #get top 3 losses
        best_lr = [self.history['epoch_lr'][-self.aneal_rate+each][0] for each in np_arr_d3]
        ideal_lr = (sum(best_lr)/len(best_lr))
        for lrs in range(len(self.scheduler.base_lrs)):
            self.scheduler.base_lrs[lrs] = ideal_lr/self.scaling # modify the LR  to best LR in model history


    def train(self,train_loader,batch_size,EPOCHS=1,test_loader=None):
        model = self.model
        history = self.history
        criterion = nn.MSELoss()
        optimizer = self.optimizer
        model.train()

        print("Starting Training of {} model".format('''GRU'''))
        # Start training loop
        for epoch in range(1, EPOCHS + 1):
            if epoch%4 ==0:
                train_loader = transform_loader()
                for lrs in range(len(self.scheduler.base_lrs)):
                    self.scheduler.base_lrs[lrs] *= self.scaling #slowly decrease the LR for scheduler to aneal
            if self.epoch % self.aneal_rate ==0 and self.epoch!=0:
                self.aneal()
            self.epoch += 1
            start_time = time.clock()
            avg_loss = 0.
            counter = 0
            for x, label in train_loader:
                counter += 1

                model.zero_grad()

                out = model(x.to(device).float())
                loss = criterion(out, label.to(device).float())
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                self.scheduler.step()
                '''
                for param_group in self.optimizer.param_groups:
                    print("Current learning rate is: {}".format(param_group['lr']))
                    #'''
                if counter % 200 == 0:
                    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                               len(train_loader),
                                                                                               avg_loss / counter))
            current_time = time.clock()
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
            print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
            history['epoch_loss'].append(avg_loss / len(train_loader))
            history['epoch_times'].append(current_time - start_time)
            lrs = []
            for each in self.scheduler.base_lrs:
                lrs.append(each)
            history['epoch_lr'].append(lrs)
            if test_loader != None:
                _o, _t, epoch_val = self.evaluate(test_loader,batch_size)
                model.train()
                history['epoch_val'].append(epoch_val)
                print('eval loss ', epoch_val)

        print("Total Training Time: {} seconds".format(str(sum(history['epoch_times']))))
        return model, history

    def evaluate(self, test_loader, batch_size, pred=False):
        model = self.model
        model.eval()
        with torch.no_grad():
            outputs = []
            targets = []
            criterion = nn.MSELoss()
            losses = 0
            i = 0
            for x, label in test_loader:
                i += 1

                out = model(x.to(device).float())
                if pred:
                    outputs.append((out.cpu().detach().numpy()))
                    targets.append((label.numpy()))
                # print("Evaluation Time: {}".format(str(time.clock() - start_time)))
                loss = criterion(out, label.to(device).float())
                losses += loss.item()
            loss = losses / i
            if pred:
                outputs, targets = np.array(outputs), np.array(targets)
                outputs, targets = outputs.reshape((-1,outputs.shape[-1])), targets.reshape((-1,targets.shape[-1]))
        return outputs, targets, loss

    def save(self,filedir): #dir/model.pth
        filepath = filedir + '/savepoint_ep' + str(self.epoch) + '.pth'
        torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                    'lr': self.lr,
                    'scheduler':self.scheduler.state_dict()
                    }, filepath)


    def load(self,filedir, epoch):
        filepath = filedir + '/savepoint_ep' + str(epoch) + '.pth'
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.epoch = checkpoint['epoch']
        self.lr = checkpoint['lr']
        self.scheduler.load_state_dict( checkpoint['scheduler'])

    def graph(self, save=False):
        from matplotlib import pyplot as plt
        lv = plt.figure()
        plt.plot(self.history['epoch_loss'])
        plt.plot(self.history['epoch_val'])
        t = plt.figure()
        plt.plot(self.history['epoch_times'])
        lr = plt.figure()
        plt.plot(self.history['epoch_lr'])

def analyse(trainer,loader):
    from matplotlib import pyplot as plt
    pred, targ, loss = trainer.evaluate(loader,0, pred=True)
    df_p = pd.DataFrame(pred)
    df_t = pd.DataFrame(targ)
    df_d = df_p - df_t
    df_d_m = df_d.mean()
    df_d_m.plot()
    df_mse_m = (df_d**2).mean()
    df_mse_m.plot()

def pred_denorm(trainer,loader,file_mean,file_std):
    from matplotlib import pyplot as plt
    pred, targ, loss = trainer.evaluate(loader, pred=True)
    df_p = pd.DataFrame(pred)
    df_t = pd.DataFrame(targ)
    df_p.columns = dat.columns
    df_t.columns = dat.columns
    dat_m = pd.read_csv(file_mean, header=None, names=['keys', 'values'], index_col='keys')
    dat_s = pd.read_csv(file_std, header=None, names=['keys', 'values'], index_col='keys')
    for i in range(len(df_p.columns)):
        df_p[df_p.columns[i]] = df_p[df_p.columns[i]] * dat_s['values'][i]  +dat_m['values'][i]
        df_t[df_t.columns[i]] = df_t[df_t.columns[i]] * dat_s['values'][i]  +dat_m['values'][i]
    return df_p, df_t

if __name__ == '__main__':
    savedir = 'PyTorch/data_analysis/saved_mod/test_7/selfval_coshot_re'
    train = trainer()
    train.boot(train_loader, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards, GPUs=[0,0,1])

    train.train(train_loader, batch_size, EPOCHS=EPOCHS, test_loader=test_loader)
    train.save(savedir)
    train.graph()

    #set to best LR from first run
    for lrs in range(len(train.scheduler.base_lrs)):
        train.scheduler.base_lrs[lrs] = 0.0008
    train.train(train_loader, batch_size, EPOCHS=25, test_loader=test_loader)
    train.save(savedir)
    train.graph()

    analyse(train,test_loader)

    batch_size = 1
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=8)
    train.drop()
    train.boot(train_loader, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards,
               GPUs=[0, 0, 1])
    train.train(train_loader, batch_size, EPOCHS=100, test_loader=test_loader)


    train = trainer()
    hists = []
    b_start=3
    d_start = 4
    for i in range(b_start,8):
        batch_size = 3 ** i
        hists.append([])
        for n in range(d_start,8):
            hidden_dim = 2**n
            train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=8)
            test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=8)

            hists[i-b_start].append([])

            train.boot(train_loader, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards,
                       GPUs=[0,0,1])
            train.train(train_loader, batch_size, EPOCHS=10, test_loader=test_loader)
            hists[i-b_start][n-d_start].append([{'batch_size': batch_size,'hidden_dim':hidden_dim},train.history])
            train.drop()
    min(hists[0][0][0][1]['epoch_loss'])
    min(hists[0][0][0][1]['epoch_val'])

    savedir = 'PyTorch/data_analysis/saved_mod/test_5'
    train = trainer()
    train.boot(train_loader, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards)
    train.load(savedir,45)

    df_p, df_t = pred_denorm(train,test_loader,file_mean='PyTorch/data/Finance_4b_4f/t_mean.csv',file_std='PyTorch/data/Finance_4b_4f/t_std.csv')
