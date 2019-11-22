import pandas as pd
import numpy as np
from matplotlib import pyplot  as plt


def show(df,cols):
    x = df.plot(y=cols[0])
    for each in cols[1:]:
        df.plot(y=each, ax=x)
