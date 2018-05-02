# How to do train valid split
Valid_ratio = 0.1
Random_state = 17
# Setting the range of the training data
From = 50000000
To = 150000000

Dtype = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32',
}

Train_kargs = {
    "parse_dates": ['click_time'],
    "dtype": Dtype, 
    "usecols": ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
}

Test_kargs = {
    "parse_dates": ['click_time'],
    "dtype": Dtype, 
    "usecols": ['ip','app','device','os', 'channel', 'click_time', 'click_id']
}

Split_kargs = {
    "test_size": Valid_ratio,
    "random_state": Random_state
}

N_train = To - From
Valid_size = 25000000
Data_path = "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/input"
Train_fname = "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/input/train.csv"
Test_fname = "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/input/test.csv"
Valid_fname = "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/input/valid_oof.csv"

Chunk_files = [
    "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/input/chunk_1.csv",
    "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/input/valid_oof.csv"
]


Target = 'is_attributed'

Predictors = ['nextClick', 'nextClick_shift', 'app', 'device', 'os', 
'channel', 'hour', 'day', 'ip_tcount', 'ip_tchan_count', 'ip_app_count', 
'ip_app_os_count', 'ip_app_os_var', 'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 
'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

Categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
