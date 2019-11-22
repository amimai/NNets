### wrangle ###
## data manipulation functions ##

import pandas
import numpy

# select columns with key names #
def filter(data,key):
    filtered = []
    for each in data.columns:
        if each.find(key)!=-1: filtered.append(each)
    return data[filtered]

# select only specific columns #
def get_cols(df,key_list):
    cols = []
    for each in df.columns:
        for key in key_list:
            if each.find(key) != -1: cols.append(each)
    return cols

# differance dataset #
def differance(data):
    ret = data.diff()
    ret = ret.iloc[1:,] # drop first row of NaN
    return ret

# differance by precentages
def p_diff(data):
    ret = data.diff()
    ret = ret/data
    ret = ret.iloc[1:, ]
    return ret

# normalize #
def mean_normalize(data):
    return (data - data.mean() ) / data.std()

def min_max_normalize(data):
    return (data-data.min())/(data.max()-data.min())

# randomly modify data by a factor 1% default
def random_wobble(data,factor=100):
    return data*(numpy.random.randint(10000-factor,10000+factor,size=data.shape)/10000.0)