import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

# Parameters and DataLoaders
input_size = 6000
output_size = 6000

batch_size = 11000
data_size = 100000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True, num_workers=4)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.fc4 = nn.Linear(input_size, output_size)
        self.fc5 = nn.Linear(input_size, output_size)
        self.fc6 = nn.Linear(input_size, output_size)
        self.fc7 = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        output = self.fc6(output)
        output = self.fc7(output)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model,device_ids=[0, 0, 1])
  #max batdch size config = [0,0,1] [bs=11000] , max speed config = 0,0,0,1 bs = [3000]




model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
