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
    print("Shape of training data: ")
    print(df.shape)
    print(df.columns)
    length = len(df)
    n_chunk = 6
    chunk_size = int((length-Valid_size)/n_chunk)
    np.random.seed(17)
    perm = np.random.permutation(np.arange(0, length))
    print("Check permutation: ")
    print(perm[: 10])

    print("Making chunked dataFrames ...")
    dfs = []
    for i in range(n_chunk):
        dfs.append(df.loc[perm[i*chunk_size: (i+1)*chunk_size]])
    dfs.append(df.loc[perm[(i+1)*chunk_size: ]])

    print("Shape of chunks: ")
    for i in range(n_chunk):
        print("Shape of chunk #" + str(i+1))
        print(dfs[i].shape)
        print(dfs[i].head(2))
    print("Shape of validation set: ")
    print(dfs[6].shape)
    print(dfs[6].head(2))

    del df
    gc.collect()

    print("Saving ...")
    for chunk_df, fname in zip(dfs, Chunk_files):
        print("Saving file to " + fname)
        chunk_df.to_csv(fname, index=False)

    gc.collect()
    print("Time usage for separating is " + str(time.time() - t_start) + " sec.")

if __name__ == "__main__":
    separate()
