import pandas as pd
import numpy as np

data_file='all_data_223k_3y_m5.csv'
data = pd.read_csv(data_file, index_col='date')
data = data*0
data = data.fillna(0)
data = data+1
data = data.cumsum()
data.to_csv('sigmoid_testset')