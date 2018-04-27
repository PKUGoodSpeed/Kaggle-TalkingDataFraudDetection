'''
Copy from https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977
'''

import os
import time
import gc
import numpy as np
import pandas as pd
from constants import *


def getExtendedFeatures(df):
    '''
    Input a raw dataFrame and
    get extended features
    '''
    assert 'click_time' in df.columns, "No click_time columns"
    print('Extracting time features ...')
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df['month'] = pd.to_datetime(df.click_time).dt.month.astype('uint8')
    df['year'] = pd.to_datetime(df.click_time).dt.year.astype('uint8')
    gc.collect()

    print('Extracting X features ... ')
    naddfeat=9
    for i in range(0, naddfeat):
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
            gp = df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            df = df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==1:
            gp = df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            df = df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==2:
            gp = df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            df = df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==3:
            gp = df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].skew().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            df = df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==4:
            gp = df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().\
                rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
            df = df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        if QQ==5:
            gp = df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount()
            df['X'+str(i)]=gp.values
        del gp
        gc.collect()

    print('doing nextClick')
    predictors=[]

    new_feature = 'nextClick'

    D=2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
        + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer= np.full(D, 3000000000, dtype=np.uint32)

    df['epochtime']= df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks= []
    for category, t in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category]-t)
        click_buffer[category]= t
    del(click_buffer)
    QQ= list(reversed(next_clicks))

    df[new_feature] = QQ
    predictors.append(new_feature)

    df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature+'_shift')

    del QQ
    gc.collect()

    print('grouping by ip-day-hour combination...')
    gp = df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    df = df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    df = df.merge(gp, on=['ip','app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    df = df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    gp = df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    df = df.merge(gp, on=['ip','day','channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    df = df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    gp = df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    df = df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    df = df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print("vars and data type: ")
    df.info()
    df['ip_tcount'] = df['ip_tcount'].astype('uint16')
    df['ip_app_count'] = df['ip_app_count'].astype('uint16')
    df['ip_app_os_count'] = df['ip_app_os_count'].astype('uint16')

    predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour'])

    for i in range(0,naddfeat):
        predictors.append('X'+str(i))
        
    print('predictors:')
    print(predictors)

    print("data shape:")
    print(df.shape)

    gc.collect()

    return df
    
if __name__ == "__main__":
    for ffrom, fto in zip(Chunk_raw_files, Chunk_ml_files):
        t_start = time.time()
        print("Extracting extra features for ML ... ")
        df = pd.read_csv(ffrom, **Train_kargs)
        df = getExtendedFeatures(df)
        print("Saving processed training chunk in " + fto)
        df.to_csv(fto, index=False)
        print("Time Usage for processing training chunk is " + str(time.time() - t_start) + " sec.")

    t_start = time.time()
    print("Extracting extra features for ML for testing set ...")
    df = pd.read_csv(Test_fname, **Train_kargs)
    df = getExtendedFeatures(df)
    print("Saving processed testng data in " + Test_ml_fname)
    df.to_csv(Test_ml_fname, index=False)
    print("Time Usage for processing testing data is " + str(time.time() - t_start) + " sec.")
