import gc
import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
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
    print(train_df.is_attributed.tolist().count(0))
    print(train_df.is_attributed.tolist().count(1))

    train_df = getExtendedFeatures(train_df)

    # train_df.fillna(0., inplace=True)

    test_df = train_df[n_valid: ]
    valid_df = train_df[n_train: n_valid]
    train_df = train_df[: n_train]

    print("train size:")
    print(train_df.shape)
    print("valid size:")
    print(valid_df.shape)
    print("test size:")
    print(test_df.shape)

    print('[{}] Start XGBoost Training'.format(time.time() - start_time))
    # Set the params(this params from Pranav kernel) for xgboost model

    params = {'eta': 0.3,
              'tree_method': "hist",
              'grow_policy': "lossguide",
              'max_leaves': 1400,  
              'max_depth': 0, 
              'subsample': 0.9, 
              'colsample_bytree': 0.7, 
              'colsample_bylevel':0.7,
              'min_child_weight':0,
              'alpha':4,
              'objective': 'binary:logistic', 
              'scale_pos_weight': 200,
              'eval_metric': 'auc', 
              'nthread':8,
              'random_state': 99, 
              'silent': True}
    train_df, cv_df = train_test_split(train_df, test_size=0.1, random_state=17)
    train = xgb.DMatrix(train_df[Predictors], train_df[Target])
    cv = xgb.DMatrix(cv_df[Predictors], cv_df[Target])
    del train_df
    del cv_df
    gc.collect()
    watchlist = [(train, 'train'), (cv, 'valid')]
    model = xgb.train(params, train, 200, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=5)
    del train
    del cv
    gc.collect()
    print('[{}] Finish XGBoost Training'.format(time.time() - start_time))

    print("Predicting...")
    test = xgb.DMatrix(test_df[Predictors])
    test_df['pred'] = model.predict(test, ntree_limit=model.best_ntree_limit)
    test_df = test_df[['click_id', 'pred']]
    del test
    gc.collect()

    print("writing...")
    test_df.to_csv(output_dir + '/test_pred.csv',index=False)
    print("done...")
    del test_df
    gc.collect()

    print("Making OOF ...")
    valid = xgb.DMatrix(valid_df[Predictors])
    valid_df['pred'] = model.predict(valid, ntree_limit=model.best_ntree_limit)
    valid_df = valid_df[['is_attributed', 'pred']]
    del valid
    gc.collect()

    print("writing...")
    valid_df.to_csv(output_dir + '/oof_pred.csv',index=False)
    print("done...")
    del valid_df
    gc.collect()

if __name__ == "__main__":
    output_dirs = [
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/xgBoost/fold_1"
    ]
    for i in range(1):
        print("Start training for fold #" + str(i+1))
        train_file = Chunk_files[i]
        valid_file = Valid_fname
        test_file = Test_fname
        output_dir = output_dirs[i]
        Shaocong(train_file, valid_file, test_file, output_dir)
        gc.collect()
