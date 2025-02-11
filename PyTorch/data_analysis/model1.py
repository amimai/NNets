import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F


class Deepnet(nn.Module):
    def __init__(self,input_shape,format,output_shape, nested = 0, droprate=.2):
        super(Deepnet, self).__init__()

        self.modlist = nn.ModuleList()
        self.format = format
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.droprate = droprate

        self.drop = nn.Dropout(p=droprate)

        self.nested = nested
        if self.nested>0:
            self.modlist_n = nn.ModuleList()

        self.build()

    def build(self):
        in_s = self.input_shape
        out_s = 0
        mod = 0
        for i in range(len(self.format)):
            for n in range(len(self.format[i])):
                # add a unit
                out_s = self.format[i][n]
                if n==0 : in_s += mod
                # if using nested structure build recursive net
                if n==1 and self.nested>0:
                    self.modlist_n.append(Deepnet(in_s,self.format,in_s, nested = self.nested-1,droprate=self.droprate))
                # print ('i = {0}, n = {1}, in = {2}, out = {3}'.format(i,n,in_s,out_s))
                self.modlist.append(nn.Linear(in_s, out_s))
                in_s = out_s
            # each new i is fed concat of all layers + z
            mod += out_s
            in_s = self.input_shape
        # finaly make rectifier unit
        # takes in formed arrays and outputs data in shape of input for addition

        if self.input_shape == self.output_shape:
            self.modlist.append(nn.Linear(mod, self.output_shape))
        else:
            self.modlist.append(nn.Linear(mod+self.input_shape, self.output_shape))

    def forward(self, z):
        conc = None
        layer = 0
        nests = 0
        #4 build bypass net
        for i in range(len(self.format)):

            #0 initialise
            if conc is None: x = z
            else: x = torch.cat((conc, z), 1)
            x = self.drop(x)

            #1 form layers
            for n in range(len(self.format[i])):
                # print('i=',i, ' n=',n)
                # print(x.size())

                # is using nested build recursive net
                if n==1 and self.nested>0:
                    x = self.modlist_n[nests](x)

                x = F.leaky_relu(self.modlist[layer](x))
                layer +=1

            #2 build conc
            if conc is None: conc = x
            else: conc = torch.cat((x, conc), 1)

        #5 subtract input and return
        if self.input_shape == self.output_shape:
            x = self.modlist[layer](conc)
            return F.leaky_relu(torch.add(x, z))
        else:
            x = F.leaky_relu(self.modlist[layer](torch.cat((conc,z),1)))
            return x


# RNN Model (Many-to-One)
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,drop_prob,batch_first=True):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size,hidden_size,num_layers,dropout=drop_prob,batch_first=batch_first)

    def init_hidden(self,batchsize,device):
        weight = next(self.parameters()).data
        hidden = weight.new(int(self.num_layers), batchsize, self.hidden_size).zero_().to(device)
        return hidden

    def forward(self, x, h):
        out, h = self.gru(x,h)
        return out, h

class lane(nn.Module):
    def __init__(self,input_dim,hidden_dim,attn_layers,backwards,drop_prob):
        super(lane, self).__init__()
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=drop_prob)

        self.backwards=backwards
        self.hidden_dim=hidden_dim

        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.attn_list = nn.ModuleList()
        self.layernorm = nn.ModuleList()
        for i in range(attn_layers):
            self.attn_list.append(torch.nn.MultiheadAttention(
                hidden_dim, 8))
            self.layernorm.append(nn.LayerNorm(hidden_dim))

    def forward(self, x):
        bs = x.shape[0]
        out = self.relu(self.lin_in(x))
        out = self.drop(out)
        out = out.transpose(0, 1)
        for i in range(len(self.attn_list)):
            tmp,attn = self.attn_list[i](out,out,out)
            out = self.layernorm[i](torch.add(tmp, out))
        out = out.transpose(0, 1)
        out = out.reshape(bs, self.backwards * self.hidden_dim)
        return out

class lane2(nn.Module):
    def __init__(self,input_dim,hidden_dim,attn_layers,backwards,drop_prob):
        super(lane2, self).__init__()
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=drop_prob)

        self.backwards=backwards
        self.hidden_dim=hidden_dim

        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.attn_list = nn.ModuleList()
        self.layernorm = nn.ModuleList()
        for i in range(attn_layers):
            self.attn_list.append(torch.nn.MultiheadAttention(
                hidden_dim, 8))
            self.layernorm.append(nn.LayerNorm(hidden_dim))

    def forward(self, x):
        #bs = x.shape[0]
        out = self.relu(self.lin_in(x))
        out = self.drop(out)
        out = out.transpose(0, 1)
        for i in range(len(self.attn_list)):
            tmp,attn = self.attn_list[i](out,out,out)
            out = self.layernorm[i](torch.add(tmp, out))
        out = out.transpose(0, 1)
        return out

class lane_merge(nn.Module):
    def __init__(self,hidden_dim,backwards,drop_prob):
        super(lane_merge, self).__init__()
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=drop_prob)

        self.backwards=backwards
        self.hidden_dim=hidden_dim

        self.attn = torch.nn.MultiheadAttention(hidden_dim, 8)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x, y):
        #bs = x.shape[0]
        out,attn = self.attn(x.transpose(0, 1),y.transpose(0, 1),y.transpose(0, 1))
        out = out.transpose(0, 1)
        out = torch.add(torch.add(x,y),out)
        out = self.layernorm(out)
        return out

class self_attn(nn.Module):
    def __init__(self,hidden_dim,backwards,drop_prob):
        super(self_attn, self).__init__()
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=drop_prob)

        self.backwards=backwards
        self.hidden_dim=hidden_dim

        self.attn = torch.nn.MultiheadAttention(hidden_dim, 8)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        #bs = x.shape[0]
        #print(x.shape)
        inx = self.drop(x).transpose(0, 1)
        out,attn = self.attn(inx,inx,inx)
        out = torch.add(inx, out)
        out = out.transpose(0, 1)
        out = self.layernorm(out)
        return out