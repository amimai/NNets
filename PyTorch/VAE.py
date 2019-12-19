# build from scratch #

import torch
import torchvision
from torchvision.utils import save_image

#hyperperams
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5

# configs
log_interval = 10
device = torch.device("cpu")

#seed fixed to create repeateable randomness :p
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#data handlers
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)) #global mean and std of mnist
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
#https://pytorch.org/tutorials/beginner/ptcheat.html
class Vae(nn.Module):
    def __init__(self):
        super(Vae, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5,stride=1,padding=2 )
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.fc1 = nn.Linear(280, 50)
        self.fc2a = nn.Linear(50, 10)
        self.fc2b = nn.Linear(50, 10)

        self.ufc1 = nn.Linear(70, 784)
        #self.upool1 = nn.MaxUnpool1d(2, stride=2)
        self.uconv1 = nn.ConvTranspose2d(10, 1, kernel_size=5, stride=1, padding=2)

    def encode(self,x):
        h1, ind = self.pool1(self.conv1(x))
        h1 = F.relu(h1)
        h1 = h1.view(-1,280)
        h1 = F.relu(self.fc1(h1))
        return self.fc2a(h1),self.fc2b(h1),ind

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self,z):
        h3 = F.relu(self.ufc1(z))
        h3 = h3.view(-1, 1, 28, 28)
        #h3 = self.uconv1(h3)
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar, ind = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1,70)), mu, logvar
'''

class Vae(nn.Module):
    def __init__(self):
        super(Vae, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc1b = nn.Linear(400, 100)
        self.fc21 = nn.Linear(100, 2)
        self.fc22 = nn.Linear(100, 2)
        self.fc3 = nn.Linear(2, 100)
        self.fc3b = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc1b(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc3b(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = Vae().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
        if batch_idx % log_interval == 0:
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
                                      recon_batch.view(batch_size_test, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main():
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 2).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

import numpy as np

n = 10
grid_x,grid_y = np.linspace(-4, 4, n), np.linspace(-4, 4, n)[::-1]



if __name__ == "__main__":
    main()
else:
    main()