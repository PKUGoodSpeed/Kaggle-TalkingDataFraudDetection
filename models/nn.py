import os
import gc
import sys
import time
import numpy as np 
import pandas as pd
sys.path.append('../utils')
from constants import *
from features import getExtendedFeatures

## Using neural network
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input

global_learning_rate = 0.0003
global_decaying_rate = 0.9
epochs = 80

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

    train_df.fillna(0., inplace=True)

    train_df.drop("app", axis=1, inplace=True)
    train_df.drop("device", axis=1, inplace=True)
    train_df.drop("os", axis=1, inplace=True)
    train_df.drop("ip", axis=1, inplace=True)
    train_df.drop("click_time", axis=1, inplace=True)
    train_df.drop("channel", axis=1, inplace=True)
    gc.collect()
    nn_predictors = ['nextClick', 'nextClick_shift', 'hour', 'day', 'month', 'year', 'ip_tcount', 
    'ip_tchan_count', 'ip_app_count', 'ip_app_os_count', 'ip_app_os_var', 'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 
    'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    for tag in nn_predictors:
        ave = train_df[tag].mean()
        std = train_df[tag].std()
        std = max(std, 1.)
        train_df[tag] = train_df[tag].apply(lambda x: (x-ave)/std)

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

    in_layer = Input((len(nn_predictors), ))
    hidden = Dropout(0.16) (Dense(64, activation='relu') (in_layer))
    hidden = Dropout(0.32) (Dense(256, activation='relu') (hidden))
    hidden = Dropout(0.64) (Dense(1024, activation='relu') (hidden))
    out_layer = Dense(1, activation='sigmoid') (hidden)
    model = Model(inputs=[in_layer], outputs=[out_layer])
    model.summary()

    model.compile(optimizer=Adam(global_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    global global_learning_rate
    global global_decaying_rate

    ## Using adaptive decaying learning rate
    def scheduler(epoch):
        global global_learning_rate
        global global_decay_rate
        if epoch%3 == 0:
            global_learning_rate *= global_decaying_rate
            print("CURRENT LEARNING RATE = " + str(global_learning_rate))
        return global_learning_rate
    change_lr = LearningRateScheduler(scheduler)
    earlystopper = EarlyStopping(patience=8, verbose=1, monitor='val_acc', mode='auto')
    if not os.path.exists('./checkpoints'):
        os.system('mkdir checkpoints')
    checkpointer = ModelCheckpoint(filepath='./checkpoints/model.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')

    model.fit(train_df[nn_predictors].values, train_df[Target].values, epochs=epochs, verbose=1, 
    validation_split=0.1, batch_size=128, class_weight={0:1, 1:40}, callbacks=[earlystopper, checkpointer, change_lr])

    del train_df
    gc.collect()

    model.load_weights('./checkpoints/model.h5')
    print("Predicting...")
    test_df['pred'] = model.predict(test_df[nn_predictors].values)
    test_df = test_df[['click_id', 'pred']]
    print("writing...")
    test_df.to_csv(output_dir + '/test_pred.csv',index=False)
    print("done...")
    del test_df

    print("Making OOF ...")
    valid_df['pred'] = model.predict(valid_df[nn_predictors].values)
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
