import pandas as pd
from pandas import DataFrame as df
import numpy as np

from torch.utils.data import Dataset

def dataprep(data_file, forward, file_data, file_truth):
    from DataProcess import wrangle as wr

    # get our massive dataset for 72 instruments #
    data = pd.read_csv(data_file, index_col='date')

    # several instruments have incomplete datasets for the last 3 years
    bad_instruments = ['FRA40', 'CHN50', 'US2000', 'USOil', 'SOYF', 'WHEATF', 'CORNF', 'EMBasket', 'JPYBasket',
                       'BTC/USD', 'BCH/USD', 'ETH/USD', 'LTC/USD', 'XRP/USD', 'CryptoMajor', 'USEquities']
    bad_cols = wr.get_cols(data, bad_instruments)

    # clean up our data and fill the gaps that are left (from market shutdown over weekend)
    data = data.drop(bad_cols, axis=1)
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')

    good_cols = wr.get_cols(data, ['bidopen','bidhigh','bidlow'])#['EUR/USD'])  # ['bidopen'])# , 'tick'
    data = data[good_cols]

    # testcode
    #test_cols = wr.get_cols(data, ['bidopenEUR/USD','bidhighEUR/USD'])
    #data = data[test_cols]

    # get the precentage differance of the data
    data = wr.p_diff(data)

    # mean norm data (used by generator)
    d_mean = data.mean()
    d_std = data.std()

    truth = data.rolling(forward).sum()
    t_mean = truth.mean()
    t_std = truth.std()

    d_msd = (data-d_mean)/d_std
    t_msd = (truth-t_mean)/t_std

    #testcode
    #d_msd = data
    #t_msd = truth

    d_msd.to_csv(file_data+'/d_msd.csv')
    t_msd.to_csv(file_truth+'/t_msd.csv')

    d_mean.to_csv(file_data + '/d_mean.csv')
    d_std.to_csv(file_data + '/d_std.csv')
    t_mean.to_csv(file_truth + '/t_mean.csv')
    t_std.to_csv(file_truth + '/t_std.csv')

class ForexDataset(Dataset):
    def __init__(self, backwards, forward, data_root=None, file_data=None,file_truth=None):
        self.samples = []

        '''#sample code for filesystem based data orgs
        for race in os.listdir(data_root):
            race_folder = os.path.join(data_root, race)

            for gender in os.listdir(race_folder):
                gender_filepath = os.path.join(race_folder, gender)

                with open(gender_filepath, 'r') as gender_file:
                    for name in gender_file.read().splitlines():
                        self.samples.append((race, gender, name))
        '''
        #extract the data
        dat = pd.read_csv(file_data, index_col='date').values
        tru = pd.read_csv(file_truth, index_col='date').values

        rolarray = []
        for t in zip(dat, tru):
            rolarray.append(t[0])
            if len(rolarray) <= forward+backwards:
                continue
            else:
                rolarray.pop(0)
                self.samples.append((rolarray[:backwards].copy(), t[1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs = self.samples[idx][0]
        targets = self.samples[idx][1]
        return inputs, targets

class ForexDataset2(Dataset):
    def __init__(self, backwards, forward, data_root=None, file_data=None,file_truth=None):
        self.samples = []

        '''#sample code for filesystem based data orgs
        for race in os.listdir(data_root):
            race_folder = os.path.join(data_root, race)

            for gender in os.listdir(race_folder):
                gender_filepath = os.path.join(race_folder, gender)

                with open(gender_filepath, 'r') as gender_file:
                    for name in gender_file.read().splitlines():
                        self.samples.append((race, gender, name))
        '''
        #extract the data
        dat = pd.read_csv(file_data, index_col='date').values
        tru = pd.read_csv(file_truth, index_col='date').values

        lookback = 90
        inputs = np.zeros((len(dat) - lookback, lookback, df.shape[1]))
        labels = np.zeros(len(dat) - lookback)

        for i in range(lookback, len(dat)):
            inputs[i - lookback] = dat[i - lookback:i]
            labels[i - lookback] = tru[i+forward, 0]
        inputs = inputs.reshape(-1, lookback, df.shape[1])
        labels = labels.reshape(-1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs = self.samples[idx][0]
        targets = self.samples[idx][1]
        return inputs, targets




if __name__ == '__main__':
    dataprep(forward=12, data_file='all_data_223k_3y_m5.csv',
             file_data='PyTorch/data/Finance_12b_12f',
             file_truth='PyTorch/data/Finance_12b_12f')

    from torch.utils.data import DataLoader
    dataset = ForexDataset(backwards=12,forward=12,
                           file_data='PyTorch/data/Finance_12b_12f/d_msd.csv',
                           file_truth='PyTorch/data/Finance_12b_12f/t_msd.csv')
    print(len(dataset))
    print(dataset[420])
    print(sum(sum(dataset[420][0])),sum(dataset[420][1]))

    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)
    for i, batch in enumerate(dataloader):
        print(i, batch)