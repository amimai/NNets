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



'''
10a model with 3 in and 3 out feeds
'''


#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lanes = 3
inputs = 4
output_dim = 3
backwards = 30

batch_size=3**5 #300, 17s, .79loss #3000, 3s, .81loss
learn_rate=1e-3
EPOCHS=3

hidden_dim=2**6
n_layers = 4
input_dim = 4




# import losses
from PyTorch.data_analysis import losses as ls

from torch.nn import functional as F


from PyTorch.data_analysis.model1 import Deepnet
from PyTorch.data_analysis.model1 import GRU
from PyTorch.data_analysis.model1 import lane2,lane_merge


class GRUNet(nn.Module):
    def __init__(self, lanes, inputs, backwards, output_dim, hidden_dim, batch_size, n_layers, drop_prob=0.40, GPUs=1):
        super(GRUNet, self).__init__()
        self.lanes = lanes

        self.input_dim = inputs
        self.output_dim = output_dim

        self.hidden_dim = hidden_dim
        self.attn_layers = 6

        self.batch_size = batch_size
        self.n_layers = n_layers

        self.backwards = backwards
        self.GPUs = GPUs
        self.drop_prob = drop_prob


        self.drop = nn.Dropout(p=drop_prob)
        self.relu = nn.LeakyReLU()

        self.lane_list = nn.ModuleList()
        self.feed_list = nn.ModuleList()
        for i in range(self.lanes):
            self.lane_list.append(lane2(inputs,hidden_dim,self.attn_layers,backwards,drop_prob))
            self.feed_list.append(lane_merge(hidden_dim,backwards,drop_prob))


        self.gru1 = GRU(int(input_dim/1), hidden_dim, n_layers, drop_prob, batch_first=True)

        self.r1_dnet = Deepnet(backwards * hidden_dim,
                               [[hidden_dim, hidden_dim],
                                [hidden_dim, hidden_dim],
                                [hidden_dim, hidden_dim]],
                               self.hidden_dim, nested=0, droprate=drop_prob)

        self.dens = nn.ModuleList()
        for i in range(self.lanes):
            self.dens.append( Deepnet(backwards * hidden_dim,
                                   [[hidden_dim, hidden_dim],
                                    [hidden_dim, hidden_dim],
                                    [hidden_dim, hidden_dim]],
                                   hidden_dim, nested=0, droprate=drop_prob))
        self.lin_outs = nn.ModuleList()
        for i in range(self.lanes):
            self.lin_outs.append(nn.Linear((hidden_dim*2)+(backwards * hidden_dim), output_dim))


    def init_hidden(self, batch_size,device):
        hidden = self.gru1.init_hidden(batch_size,device)
        return hidden

    def forward(self, x,y,z, h):
        bs=x.shape[0]
        #print(x.shape)
        out = self.lane_list[2](z)
        t1 = self.lane_list[1](y)
        t0 = self.lane_list[0](x)
        out = self.feed_list[2](t1,out)
        out = self.feed_list[1](t0,out)

        out = out.reshape(bs, self.backwards * self.hidden_dim)
        #print(out.shape)


        #print(out.shape)
        #out = self.r1_dnet(out)

        tmp = self.r1_dnet(out)

        return self.lin_outs[0](torch.cat([tmp,self.dens[0](out),out],1)),\
               self.lin_outs[1](torch.cat([tmp,self.dens[1](out),out],1)),\
               self.lin_outs[2](torch.cat([tmp,self.dens[2](out),out],1)), h

        #return self.lin_outs[0](out),self.lin_outs[0](out),self.lin_outs[0](out),h

class trainer:
    def __init__(self, t_x=None,t_y=None,batch_size=None,model=None, history=None, optimizer = None, scheduler=None, lr=1e-3):
        self.model = model
        self.history = history
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.epoch = 0
        self.scaling = .8 #how much LR decreases every micro cycle
        self.aneal_rate = 3**3  #how often we update LR to best LR
        self.coshot_cycle = int(5825)
        self.t_x = t_x
        self.t_y = t_y
        self.trainloader = None
        self.batch_size = batch_size

        self.bestmodel = None
        self.age = 0
        self.age_factor=1.
        self.loss = float('inf')
        self.losscount = 300
        self.lossadjust = .98
        self.lossreverse = (1-self.lossadjust)*(1+self.lossadjust**(self.losscount**self.lossadjust))
        self.losses = [float('inf')]*self.losscount

        self.raw_dump = []
        self.dumper = False


    def drop(self):
        self.model=None
        self.optimizer=None
        self.history=None
        self.epoch=0

    def boot(self,batch_size,hidden_dim=256,output_dim=168,n_layers=2,backwards=48, GPUs=[0], trainloader=None,lanes=3):
        self.batch_size = batch_size
        if trainloader is None:
            self.transform_loader()
            train_loader = self.trainloader
        else:
            self.trainloader = trainloader
            train_loader = trainloader
        if self.model is None:
            input_dim = next(iter(train_loader))[0][0].shape[1] # lanes, inputs, backwards, output_dim, hidden_dim, batch_size, n_layers
            self.model = GRUNet(lanes, input_dim, backwards, output_dim, hidden_dim, batch_size, n_layers, GPUs=len(GPUs))
        if self.history is None:
            self.history = {'epoch_loss':   [],
                            'epoch_times':  [],
                            'epoch_val':    [],
                            'epoch_lr':     []}
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) #torch.optim.SGD(self.model.parameters(), lr=self.lr,momentum=0.9,nesterov=True)
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.coshot_cycle, T_mult=1, eta_min=0, last_epoch=-1)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model, device_ids=GPUs)
        self.model.to(device)
        #summary(self.model, (backwards, input_dim))

    def transform_loader(self):
        'transforms data in loader by a random'
        self.trainloader = None
        n_x = self.t_x + (((np.random.rand(*self.t_x.shape) - .5) / 1000))
        n_y = self.t_y + (((np.random.rand(*self.t_y.shape) - .5) / 1000))

        n_train_data = TensorDataset(torch.from_numpy(n_x), torch.from_numpy(n_y))
        n_x = None
        n_y = None
        self.trainloader = DataLoader(n_train_data, shuffle=True, batch_size=self.batch_size, drop_last=True, num_workers=16)

    def moving_average(self,data_set, periods=3):
        weights = np.ones(periods) / periods
        return np.convolve(data_set, weights, mode='valid')

    def aneal(self):
        losses = np.array(self.history['epoch_loss'][-self.aneal_rate:])
        np_arr_d3 = self.moving_average(np.diff(losses, 1)/losses[:-1],periods=3).argsort()[:3] #get top 3 losses most -ve precentile
        best_lr = [self.history['epoch_lr'][-self.aneal_rate+each][0] for each in np_arr_d3]
        ideal_lr = (sum(best_lr)/len(best_lr))
        for lrs in range(len(self.scheduler.base_lrs)):
            self.scheduler.base_lrs[lrs] = ideal_lr/self.scaling # modify the LR  to best LR in model history

    def re_init_opts(self,opt):
        self.optimizer = opt
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.coshot_cycle, T_mult=1, eta_min=0, last_epoch=-1)

    def train(self,batch_size,EPOCHS=1,test_loader=None,transform = False):
        model = self.model
        history = self.history
        losses = [nn.MSELoss()] #,nn.L1Loss(),ls.scaled_error_d()ls.L3_MSE_2()#ls.scaled_error_d()##ls.L3_STD_MSE()#nn.MSELoss()
        optimizer = self.optimizer
        model.train()


        print("Starting Training of {} model".format('''GRU'''))
        # Start training loop
        for epoch in range(1, EPOCHS + 1):
            if epoch%1==0:
                if transform:
                    self.transform_loader()
                for lrs in range(len(self.scheduler.base_lrs)):
                    self.scheduler.base_lrs[lrs] *= self.scaling #slowly decrease the LR for scheduler to aneal
            if self.epoch % self.aneal_rate ==0 and self.epoch!=0:
                self.aneal()
            train_loader = self.trainloader
            self.epoch += 1
            start_time = time.clock()
            avg_loss = 0.
            counter = 0

            h = model.module.init_hidden(batch_size,device)

            for x,y,z,t0,t1,t2 in train_loader:
                counter += 1

                model.zero_grad()

                h = h.data
                o0,o1,o2, h = model(x.to(device).float(),y.to(device).float(),z.to(device).float(),h)
                for i in range(len(losses)):
                    tmp_loss = torch.add(torch.add(losses[i](o0,t0.to(device).float()),losses[i](o1,t1.to(device).float())),losses[i](o2,t2.to(device).float()))
                    if i == 0: loss = tmp_loss
                    else: loss = torch.add(loss,tmp_loss)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                self.scheduler.step()
                '''
                for param_group in self.optimizer.param_groups:
                    print("Current learning rate is: {}".format(param_group['lr']))
                    #'''

                self.losses.append(loss.item())
                self.losses.pop(0)
                roll_loss = sum(self.losses)*self.lossreverse #loss is calculated using ema
                self.losses = [i*self.lossadjust for i in self.losses] #lower weight of old losses
                if roll_loss<self.loss:
                    self.loss = roll_loss
                    self.bestmodel = self.model.state_dict()
                    self.age = 0
                    self.age_factor=1.
                    self.dumper=False
                    #print('best loss is now :',roll_loss)
                else:
                    self.age += 1
                    if self.age>self.losscount*self.age_factor:
                        if not self.dumper: #store loss failures
                            self.dumper = True
                            self.raw_dump.append((self.epoch, counter, self.loss))
                        #print(self.loss, ' pre')
                        self.loss *=1.10**self.age_factor # increase the loss threshold if trapped in blind alley
                        #print(self.loss,' mid')
                        self.loss += roll_loss*0.01
                        #print(self.loss, ' post')
                        self.age = 0
                        self.age_factor *= 1.5
                        self.model.load_state_dict(self.bestmodel)

                if counter % 20000 == 0:
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
                del _o, _t
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
            h = model.module.init_hidden(batch_size,device)
            for x,y,z,t0,t1,t2 in test_loader:
                i += 1

                h = h.data
                o0,o1,o2, h = model(x.to(device).float(),y.to(device).float(),z.to(device).float(),h)
                if pred:
                    outputs.append((o0.cpu().detach().numpy(),o1.cpu().detach().numpy(),o2.cpu().detach().numpy()))
                    targets.append((t0.numpy(),t1.numpy(),t2.numpy()))
                # print("Evaluation Time: {}".format(str(time.clock() - start_time)))
                loss = torch.add(torch.add(criterion(o0,t0.to(device).float()),criterion(o1,t1.to(device).float())),criterion(o2,t2.to(device).float()))
                losses += loss.item()
            loss = losses / i
            if pred:
                outputs, targets = np.array(outputs), np.array(targets)
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
    df_lwe = (df_d**2).mean(axis=1)
    df_lwe.plot()

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
    #from PyTorch.data_analysis.datastore import train_data, test_data
    #from PyTorch.data_analysis.test_model9a import *

    savedir = 'PyTorch/data_analysis/saved_mod/test_9/s'
    train = trainer()
    train.boot(batch_size, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards,
               GPUs=[0], trainloader=train_loader)
    for i in range(5):
        train.train(batch_size, EPOCHS=3, test_loader=test_loader)
        train.save(savedir)
    train.graph()

    #set to best LR from first run
    for lrs in range(len(train.scheduler.base_lrs)):
        train.scheduler.base_lrs[lrs] = 0.0008
    train.train(batch_size, EPOCHS=EPOCHS, test_loader=test_loader)
    train.save(savedir)
    train.graph()

    analyse(train,test_loader)

    batch_size = 1
    train.boot(batch_size, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards,GPUs = [0])
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=16)
    train.drop()
    train.boot(batch_size, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards,GPUs = [0])
    train.load(savedir,EPOCHS*2)
    train.train(batch_size, EPOCHS=EPOCHS, test_loader=test_loader)



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

    import numpy as np
    dat = np.array([[i, i] for i in range(100)])
    backwards = 5
    spacing = 10
    inputs = np.zeros((len(dat) - backwards*spacing, backwards, dat.shape[-1]))
    labels = np.zeros((len(dat) - backwards*spacing, dat.shape[-1]))
    for i in range(backwards*spacing, len(dat)):
        inputs[i - backwards * spacing] = dat[i - backwards * spacing:i][::spacing]
        labels[i - backwards * spacing] = dat[i:i+1]

    # binarized check
    pred, targ, loss = train.evaluate(test_loader, 0, pred=True)
    dtb = (pd.DataFrame(targ) > 0)
    dpb = (pd.DataFrame(pred) > 0)
    db_df = dpb ^ dtb
    db_df.mean(axis=0).plot()
    (db_df.mean(axis=1).rolling(200).sum() / 200).plot()

    t = 3
    df_p = pd.DataFrame(pred)
    df_t = pd.DataFrame(targ)
    df_d = df_p - df_t
    df_t.iloc[:, t].plot()
    df_p.iloc[:, t].plot()
    (df_p.iloc[:, t] + df_p.iloc[:, t + 1]).plot() #highs
    (df_p.iloc[:, t] + df_p.iloc[:, t + 2]).plot() #lows
    df_lwe = (df_d.iloc[:, :3] ** 2).mean(axis=1)
    df_lwe.plot()

    negative = ((db_df * df_t) ** 2) ** .5
    positive = (((dpb == dtb) * df_t) ** 2) ** .5
    pl_diff = positive-negative
    pl_diff.iloc[:,::4].mean(axis=1).plot()
    (pl_diff.iloc[:,::4].mean(axis=1).rolling(200).sum() / 200).plot()
    print(pl_diff.mean().mean()/((df_t**2)**.5).mean().mean())
    print(pl_diff.iloc[:,::4].cumsum().mean().mean())


    pred, targ, loss = train.evaluate(test_loader, 0, pred=True)
    df_p = pd.DataFrame(pred)
    df_t = pd.DataFrame(targ)
    df_d = df_p - df_t
    tmp = []
    t=0
    for i in range(15):
        thresh = .05 + i / 10
        if tmp is None:
            tmp = (((df_p.iloc[:, t] < thresh) & (df_p.iloc[:, t] > thresh / 2)) * df_t.iloc[:, t]
                   - ((df_p.iloc[:, t] > -thresh) & (df_p.iloc[:, t] < -thresh / 2) * df_t.iloc[:, t])).cumsum()
        else:
            tmp.append((((df_p.iloc[:, t] < thresh) & (df_p.iloc[:, t] > thresh / 2)) * df_t.iloc[:, t]
                        - ((df_p.iloc[:, t] > -thresh) & (df_p.iloc[:, t] < -thresh / 2)) * df_t.iloc[:,t]).cumsum())
        tmp[-1].plot()
    print([each.mean() for each in tmp]) #mean returns
    print([(each*(each<0)).mean() for each in tmp]) #mean downdraw
    print(np.array([each.mean() for each in tmp])/-np.array([(each*(each<0)).mean() for each in tmp])) #mean profit per loss

    #use both range bands and actual pred to calculate decision
    #reliable on t3 and t6
    t = 6
    up = (((df_p.iloc[:, t] > .2)) * df_t.iloc[:, t])
    down = (((df_p.iloc[:, t] < -.2)) * -1 * df_t.iloc[:, t])
    (up + down).cumsum().plot()
    up = (((((df_p.iloc[:, t + 1] ** 2) < (df_p.iloc[:, t + 2] ** 6)))) * df_t.iloc[:, t])
    down = (((((df_p.iloc[:, t + 1] ** 2) > (df_p.iloc[:, t + 2] ** 6)))) * -1 * df_t.iloc[:, t])
    (up + down).cumsum().plot()



    del pred,targ,loss,df_p,df_t,df_d,tmp,dtb,dpb,db_df,negative,positive,pl_diff,df_lwe

    #allows continuation
    import imp
    from PyTorch.data_analysis import test_model9a
    from PyTorch.data_analysis import losses as ls
    imp.reload(ls)
    imp.reload(test_model9a)

    from PyTorch.data_analysis.test_model10a import *
    savedir = 'PyTorch/data_analysis/saved_mod/test_10/a'
    train = trainer()
    train.boot(batch_size, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards,
               GPUs=[0], trainloader=dataloader)
    #train.load(savedir,24)
    for i in range(5):
        train.train(batch_size, EPOCHS=3, test_loader=train_dataloader)
        train.save(savedir)
    train.graph()
