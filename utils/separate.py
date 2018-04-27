"""
This script separate the training data frame into 4 parts, so than we can make training one by one
"""
import os
import gc
import sys
import time
import numpy as np
import pandas as pd
from constants import *

def separate():
    t_start = time.time()
    print("Loading training data ...")
    df = pd.read_csv(Train_fname, **Train_kargs)
    print(df.columns)
    length = len(df)
    chunk_size = int(length/4)
    perm = np.random.permutation(np.arange(0, length))
    df_1 = df.loc[perm[: chunk_size]]
    df_2 = df.loc[perm[chunk_size: 2*chunk_size]]
    df_3 = df.loc[perm[2*chunk_size: 3*chunk_size]]
    df_4 = df.loc[perm[3*chunk_size: ]]
    
    print("Shape of chunks: ")
    print(df_1.shape)
    df_1.head(2)
    print(df_2.shape)
    df_2.head(2)
    print(df_3.shape)
    df_3.head(2)
    print(df_4.shape)
    df_4.head(2)
    assert len(df) == len(df_1) + len(df_2) + len(df_3) + len(df_4)
    del df
    gc.colect()

    print("Saving ...")
    print("Chunk #1 saved in " + Data_path + '/chunk_1.csv!')
    df_1.to_csv(Data_path + '/chunk_1.csv', index=False)
    print("Chunk #2 saved in " + Data_path + '/chunk_2.csv!')
    df_2.to_csv(Data_path + '/chunk_2.csv', index=False)
    print("Chunk #3 saved in " + Data_path + '/chunk_3.csv!')
    df_3.to_csv(Data_path + '/chunk_3.csv', index=False)
    print("Chunk #4 saved in " + Data_path + '/chunk_4.csv!')
    df_4.to_csv(Data_path + '/chunk_4.csv', index=False)

    gc.colect()
    print("Time usage for separating is " + str(time.time() - t_start) + " sec.")

if __name__ == "__main__":
    separate()
