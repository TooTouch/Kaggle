import pandas as pd
import numpy as np
import os

def loaddata():
    root_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))
    data_dir = root_dir + '/dataset'
    train = pd.read_csv(data_dir + '/train.csv', index_col=0)
    test = pd.read_csv(data_dir + '/test.csv', index_col=0)

    return train, test

