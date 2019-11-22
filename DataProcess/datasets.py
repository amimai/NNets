### datasets ###
## builds datasets for ML ##

import numpy as np


def to_dataset(data):
    return data.loc[:,data.columns].values

def to_series(data,lookback):
    values = to_dataset(data)
    t = []
    for i in range(len(values)-lookback):
            t.append(values[i:i+lookback])
    return np.array(t)

# give random train test and val dataset in tuples #
def random_TTV(data, truth, pTest, pVal):
    total = len(data)
    test = int(total*pTest)
    val = int(total * pVal)
    indices = np.random.choice(data.shape[0], total, replace=False)
    return ((data[indices[test+val:]],      truth[indices[test+val:]]),
            (data[indices[val:test+val]],   truth[indices[val:test+val]]),
            (data[indices[:val]],           truth[indices[:val]]))

def seq_stateful(data, truth, pTest, pVal):
    total = len(data)
    test = int(total * pTest)
    val = int(total * pVal)
    return ((data[test + val:], truth[test + val:]),
            (data[val:test + val], truth[val:test + val]),
            (data[:val], truth[:val]))

