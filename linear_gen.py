####################################################
# generating linear test data for quick prototyping#
####################################################
# goals:
# make custom dataset
# save to file
# 
# 
# 

import numpy as np
from random import randrange
from math import cos

def gen_data(amm):
    x = np.zeros((amm, 1))
    for i in range(amm):
        x[i] = [cos(i+(randrange(-10,10)/10.))]  # norm,norm,geometric,fibinacci,exponential
    return x


print(gen_data(5))