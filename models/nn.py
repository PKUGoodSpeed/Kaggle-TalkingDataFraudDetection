import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import gc

## Keras utils
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

## Label encoding
from sklearn.preprocessing import LabelEncoder

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

    print('hour, day, wday....')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')
    print('grouping by ip-day-hour combination....')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    del gp; gc.collect()
    print('group by ip-app combination....')
    gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    del gp; gc.collect()
    print('group by ip-app-os combination....')
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp; gc.collect()
    print("vars and data type....")
    train_df['qty'] = train_df['qty'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
    print("label encoding....")

    train_df[['app','device','os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)
    print ('final part of preparation....')
    
    test_df = train_df[n_valid: ]
    valid_df = train_df[n_train: n_valid]
    train_df = train_df[: n_train]

    print("train size:")
    print(train_df.shape)
    print("valid size:")
    print(valid_df.shape)
    print("test size:")
    print(test_df.shape)

    y_train = train_df['is_attributed'].values
    train_df.drop(['click_id', 'click_time','ip','is_attributed'], 1, inplace=True)

    print ('neural network....')

    max_app = np.max([train_df['app'].max(), valid_df['app'].max(), test_df['app'].max()])+1
    max_ch = np.max([train_df['channel'].max(), valid_df['channel'].max(), test_df['channel'].max()])+1
    max_dev = np.max([train_df['device'].max(), valid_df['device'].max(), test_df['device'].max()])+1
    max_os = np.max([train_df['os'].max(), valid_df['os'].max(), test_df['os'].max()])+1
    max_h = np.max([train_df['hour'].max(), valid_df['hour'].max(), test_df['hour'].max()])+1
    max_d = np.max([train_df['day'].max(), valid_df['day'].max(), test_df['day'].max()])+1
    max_wd = np.max([train_df['wday'].max(), valid_df['wday'].max(), test_df['wday'].max()])+1
    max_qty = np.max([train_df['qty'].max(), valid_df['qty'].max(), test_df['qty'].max()])+1
    max_c1 = np.max([train_df['ip_app_count'].max(), valid_df['ip_app_count'].max(), test_df['ip_app_count'].max()])+1
    max_c2 = np.max([train_df['ip_app_os_count'].max(), valid_df['ip_app_os_count'].max(), test_df['ip_app_os_count'].max()])+1

    def get_keras_data(dataset):
        X = {
            'app': np.array(dataset.app),
            'ch': np.array(dataset.channel),
            'dev': np.array(dataset.device),
            'os': np.array(dataset.os),
            'h': np.array(dataset.hour),
            'd': np.array(dataset.day),
            'wd': np.array(dataset.wday),
            'qty': np.array(dataset.qty),
            'c1': np.array(dataset.ip_app_count),
            'c2': np.array(dataset.ip_app_os_count)
        }
        return X
    train_df = get_keras_data(train_df)

    emb_n = 50
    dense_n = 1000
    in_app = Input(shape=[1], name = 'app')
    emb_app = Embedding(max_app, emb_n)(in_app)
    in_ch = Input(shape=[1], name = 'ch')
    emb_ch = Embedding(max_ch, emb_n)(in_ch)
    in_dev = Input(shape=[1], name = 'dev')
    emb_dev = Embedding(max_dev, emb_n)(in_dev)
    in_os = Input(shape=[1], name = 'os')
    emb_os = Embedding(max_os, emb_n)(in_os)
    in_h = Input(shape=[1], name = 'h')
    emb_h = Embedding(max_h, emb_n)(in_h) 
    in_d = Input(shape=[1], name = 'd')
    emb_d = Embedding(max_d, emb_n)(in_d) 
    in_wd = Input(shape=[1], name = 'wd')
    emb_wd = Embedding(max_wd, emb_n)(in_wd) 
    in_qty = Input(shape=[1], name = 'qty')
    emb_qty = Embedding(max_qty, emb_n)(in_qty) 
    in_c1 = Input(shape=[1], name = 'c1')
    emb_c1 = Embedding(max_c1, emb_n)(in_c1) 
    in_c2 = Input(shape=[1], name = 'c2')
    emb_c2 = Embedding(max_c2, emb_n)(in_c2) 
    fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h), 
                     (emb_d), (emb_wd), (emb_qty), (emb_c1), (emb_c2)])
    s_dout = SpatialDropout1D(0.2)(fe)
    fl1 = Flatten()(s_dout)
    conv = Conv1D(100, kernel_size=4, strides=1, padding='same')(s_dout)
    fl2 = Flatten()(conv)
    concat = concatenate([(fl1), (fl2)])
    x = Dropout(0.2)(Dense(dense_n,activation='relu')(concat))
    x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
    outp = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_d,in_wd,in_qty,in_c1,in_c2], outputs=outp)

    batch_size = 256
    epochs = 2
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(list(train_df)[0]) / batch_size) * epochs
    lr_init, lr_fin = 0.002, 0.0002
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=0.002, decay=lr_decay)
    model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

    model.summary()

    class_weight = {0:1., 1:160.} # magic
    model.fit(train_df, y_train, batch_size=batch_size, epochs=2, class_weight=class_weight, shuffle=True, verbose=2)
    del train_df, y_train; gc.collect()
    model.save_weights('./imbalanced_data.h5')

    print("Predicting...")
    test_df['pred'] = model.predict(get_keras_data(test_df), batch_size=batch_size, verbose=2)
    test_df = test_df[['click_id', 'pred']]
    print("writing...")
    test_df.to_csv(output_dir + '/test_pred.csv',index=False)
    print("done...")
    del test_df
    
    print("Making OOF ...")
    valid_df['pred'] = bst.predict(get_keras_data(valid_df), batch_size=batch_size, verbose=2)
    valid_df = valid_df[['is_attributed', 'pred']]
    print("writing...")
    valid_df.to_csv(output_dir + '/oof_pred.csv',index=False)
    print("done...")
    del valid_df

if __name__ == "__main__":
    output_dirs = [
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/neuralNetwork/fold_1",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/neuralNetwork/fold_2",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/neuralNetwork/fold_3",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/neuralNetwork/fold_4",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/neuralNetwork/fold_5",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/neuralNetwork/fold_6",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/neuralNetwork/fold_7",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/neuralNetwork/fold_8",
        "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/output/neuralNetwork/fold_9"
    ]
    for i in range(9):
        print("Start training for fold #" + str(i+1))
        train_file = Chunk_files[i]
        valid_file = Valid_fname
        test_file = Test_fname
        output_dir = output_dirs[i]
        Shaocong(train_file, valid_file, test_file, output_dir)
        gc.collect()