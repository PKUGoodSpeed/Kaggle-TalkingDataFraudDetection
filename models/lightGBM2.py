"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels
This kernel gives LB: 0.977 when the parameter 'debug' below is set to 0 but this implementation requires a machine with ~32 GB of memory
"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
import os
import sys
sys.path.append('../utils')
from constants import *

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'scale_pos_weight': 20,
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1, bst1.best_iteration)

def Shaocong(train_file, valid_file, test_file, output_dir):
    print("Make Preparations ...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('loading train data ...')
    train_df = pd.read_csv(train_file, nrows=25000000, **Train_kargs)
    n_train = len(train_df)

    print('loading validation data ...')
    train_df = train_df.append(pd.read_csv(valid_file, **Train_kargs))
    n_valid = len(train_df)

    print('loading test data ...')
    train_df = train_df.append(pd.read_csv(test_file, **Test_kargs))
    gc.collect()

    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

    gc.collect()

    naddfeat=9
    for i in range(0,naddfeat):
        if i==0: selcols=['ip', 'channel']; QQ=4;
        if i==1: selcols=['ip', 'device', 'os', 'app']; QQ=5;
        if i==2: selcols=['ip', 'day', 'hour']; QQ=4;
        if i==3: selcols=['ip', 'app']; QQ=4;
        if i==4: selcols=['ip', 'app', 'os']; QQ=4;
        if i==5: selcols=['ip', 'device']; QQ=4;
        if i==6: selcols=['app', 'channel']; QQ=4;
        if i==7: selcols=['ip', 'os']; QQ=5;
        if i==8: selcols=['ip', 'device', 'os', 'app']; QQ=4;
        print('selcols',selcols,'QQ',QQ)

        if QQ==0:
            gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==1:
            gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==2:
            gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==3:
            gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].skew().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==4:
            gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==5:
            gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount()
            train_df['X'+str(i)]=gp.values

        del gp
        gc.collect()

    print('doing nextClick')
    predictors=[]

    new_feature = 'nextClick'
    D=2**26
    train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
        + "_" + train_df['os'].astype(str)).apply(hash) % D
    click_buffer= np.full(D, 3000000000, dtype=np.uint32)

    train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks= []
    for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
        next_clicks.append(click_buffer[category]-t)
        click_buffer[category]= t
    del(click_buffer)
    QQ= list(reversed(next_clicks))

    train_df[new_feature] = QQ
    predictors.append(new_feature)

    train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature+'_shift')

    del QQ
    gc.collect()

    print('grouping by ip-day-hour combination...')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print("vars and data type: ")
    train_df.info()
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    target = 'is_attributed'
    predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour'])
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    for i in range(0,naddfeat):
        predictors.append('X'+str(i))

    print('predictors',predictors)

    test_df = train_df[n_valid: ]
    valid_df = train_df[n_train: n_valid]
    train_df = train_df[: n_train]
    train_df, cv_df = train_test_split(train_df, test_size=0.1)

    print("train size:")
    print(train_df.shape)
    print("cv size:")
    print(cv_df.shape)
    print("valid size:")
    print(valid_df.shape)
    print("test size:")
    print(test_df.shape)

    gc.collect()

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.20,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 
    }
    (bst, best_iteration) = lgb_modelfit_nocv(params, 
                            train_df, 
                            cv_df, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del cv_df
    gc.collect()

    print("Predicting...")
    test_df['pred'] = bst.predict(test_df[predictors], num_iteration=best_iteration)
    test_df = test_df[['click_id', 'pred']]
    print("writing...")
    test_df.to_csv(output_dir + '/test_pred.csv',index=False)
    print("done...")
    del test_df
    
    print("Making OOF ...")
    valid_df['pred'] = bst.predict(valid_df[predictors], num_iteration=best_iteration)
    valid_df = valid_df[['is_attributed', 'pred']]
    print("writing...")
    valid_df.to_csv(output_dir + '/oof_pred.csv',index=False)
    print("done...")
    del valid_df

if __name__ == "__main__":
    output_dirs = [
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/lightGBM_2/fold_1",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/lightGBM_2/fold_2",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/lightGBM_2/fold_3",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/lightGBM_2/fold_4",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/lightGBM_2/fold_5",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/lightGBM_2/fold_6",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/lightGBM_2/fold_7",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/lightGBM_2/fold_8",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/lightGBM_2/fold_9"
    ]
    for i in range(9):
        print("Start training for fold #" + str(i+1))
        train_file = Chunk_files[i]
        valid_file = Valid_fname
        test_file = Test_fname
        output_dir = output_dirs[i]
        Shaocong(train_file, valid_file, test_file, output_dir)
        gc.collect()
