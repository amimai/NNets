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



now = time.time()
backwards = 128
spacing = 120 #spacing used to determine the space between each datapoint grouping data between

inputs = np.load('PyTorch/data/built_f1_f120/inputs_ns_EUR_USDGBP_JPYAUD_CAD.npy')
labels = np.load('PyTorch/data/built_f1_f120/labels_ns_EUR_USDGBP_JPYAUD_CAD.npy')

labels = labels.clip(min=-2.,max=2.)
labels = np.delete(labels, np.arange(3,labels.size,4)).reshape(*labels.shape[:-1],-1) #remove ticks from targets
#labels = labels[:,::3] # only look at opens

train_x = []
train_y = []
test_x = {}
test_y = {}

print('splitting test labels', time.time()-now)
# datasplit
test_portion = int(len(inputs)*.01)
if len(train_x) == 0:
    train_x = inputs[:-test_portion]
    train_y = labels[:-test_portion]
else:
    train_x = np.concatenate((train_x, inputs[:-test_portion]))
    train_y = np.concatenate((train_y, labels[:-test_portion]))
test_x = (inputs[-test_portion:])
test_y = (labels[-test_portion:])
del inputs
del labels

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
del train_x
del train_y

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
del test_x
del test_y