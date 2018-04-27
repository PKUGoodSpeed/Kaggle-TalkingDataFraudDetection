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
    "skiprows": range(1, From), 
    "nrows": To-From,
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

Train_fname = "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/input/train.csv"
Test_fname = "/home/zebo/git/myRep/Kaggle/Kaggle-TalkingDataFraudDetection/input/test.csv"
