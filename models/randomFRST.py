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

def Shaocong(train_file, valid_file, test_file, output_dir)


rf = RandomForestClassifier(n_estimators=13, max_depth=13, random_state=13,verbose=2, oob_score=VALIDATE)
rf.fit(df_train, Learning_Y)


predictions = rf.predict_proba(df_test)