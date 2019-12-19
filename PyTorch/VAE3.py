from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class deepnet(nn.Module):
    def __init__(self,format,input_shape,output_shape, nested = 0):
        super(deepnet, self).__init__()

        self.modlist = nn.ModuleList()
        self.format = format
        self.input_shape = input_shape
        self.output_shape = output_shape

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
                    self.modlist_n.append(deepnet(self.format,in_s,in_s, nested = self.nested-1))
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
            x = F.leaky_relu(self.modlist[layer](conc))
            return F.leaky_relu(torch.add(x, z))
        else:
            x = F.leaky_relu(self.modlist[layer](torch.cat((conc,z),1)))
            return x





class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 750)
        self.fc1b = nn.Linear(750, 300)
        self.fc21 = nn.Linear(300, 3)
        self.fc22 = nn.Linear(300, 3)
        self.fc3 = nn.Linear(3, 300)
        self.fc3c = nn.Linear(300, 750)
        self.fc4 = nn.Linear(750, 784)

        self.d1 = deepnet([[64,64],[64,64],[64,64],[64,64]],300,300, nested=1)
        self.d2 = deepnet([[64,64],[64,64],[64,64],[64,64]],3,300, nested=1)

        self.d3 = deepnet([[128, 128, 128], [128, 128, 128], [128, 128, 128], [128, 128, 128]], 750,750, nested=1)
        self.d4 = deepnet([[128, 128, 128], [128, 128, 128], [128, 128, 128], [128, 128, 128]], 750,750, nested=1)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        h1 = self.d3(h1)
        h1 = F.leaky_relu(self.fc1b(h1))
        h1 = self.d1(h1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.d2(z)
        h3 = F.leaky_relu(self.fc3c(h3))
        h3 = self.d4(h3)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
summary(model,(64,1,28,28))
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 3).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

    import numpy as np

    with torch.no_grad():
        sp_dat = []
        for i, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            mu,logvar = model.encode(data.view(-1,784))
            rep = model.reparameterize(mu,logvar)
            sp_dat.append((rep,targets))

    x = []
    y = []
    z = []
    c = []
    for i in range(5):
        data = sp_dat[i]  # [tensors,targets]
        e_val = data[0].numpy()
        truth = data[1].numpy()

        for i in range(len(e_val)):
            w,e,r = e_val[i]
            c.append(truth[i])
            x.append(w)
            y.append(e)
            z.append(r)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()





