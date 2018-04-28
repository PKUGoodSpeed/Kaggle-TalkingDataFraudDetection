import os
import gc
import sys
import time
import numpy as np 
import pandas as pd
#Use RandomForest
from sklearn.ensemble import RandomForestClassifier
sys.path.append('../utils')
from constants import *
from features import getExtendedFeatures

def Shaocong(train_file, valid_file, test_file, output_dir):
    print("Make Preparations ...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('loading train data ...')
    train_df = pd.read_csv(train_file, **Train_kargs)
    n_train = len(train_df)

    print('loading validation data ...')
    train_df = train_df.append(pd.read_csv(valid_file, **Train_kargs))
    n_valid = len(train_df)

    print('loading test data ...')
    train_df = train_df.append(pd.read_csv(test_file, **Test_kargs))
    gc.collect()

    train_df = getExtendedFeatures(train_df)

    train_df.fillna(0., inplacce=True)

    test_df = train_df[n_valid: ]
    valid_df = train_df[n_train: n_valid]
    train_df = train_df[: n_train]

    print("train size:")
    print(train_df.shape)
    print("valid size:")
    print(valid_df.shape)
    print("test size:")
    print(test_df.shape)

    gc.collect()

    rf = RandomForestClassifier(n_estimators=17, max_depth=17, random_state=17,verbose=2, oob_score=True)
    rf.fit(train_df[Predictors], train_df[Target])

    del train_df
    gc.collect()

    print("Predicting...")
    test_df['pred'] = rf.predict_proba(test_df[Predictors])[:, 1]
    test_df = test_df[['click_id', 'pred']]
    print("writing...")
    test_df.to_csv(output_dir + '/test_pred.csv',index=False)
    print("done...")
    del test_df

    print("Making OOF ...")
    valid_df['pred'] = rf.predict_proba(valid_df[Predictors])[:, 1]
    valid_df = valid_df[['is_attributed', 'pred']]
    print("writing...")
    valid_df.to_csv(output_dir + '/oof_pred.csv',index=False)
    print("done...")
    del valid_df

if __name__ == "__main__":
    output_dirs = [
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/randomFRST/fold_1",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/randomFRST/fold_2",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/randomFRST/fold_3",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/randomFRST/fold_4",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/randomFRST/fold_5",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/randomFRST/fold_6"
    ]
    for i in range(6):
        print("Start training for fold #" + str(i+1))
        train_file = Chunk_files[i]
        valid_file = Valid_fname
        test_file = Test_fname
        output_dir = output_dirs[i]
        Shaocong(train_file, valid_file, test_file, output_dir)
        gc.collect()