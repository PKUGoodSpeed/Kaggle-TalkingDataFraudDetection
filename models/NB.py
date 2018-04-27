import os
import io
import sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
sys.path.append('../utils')
from ml_process import getExtendedFeatures

if __name__ == "__main__":
    train, cv, valid, test, predictors = getExtendedFeatures()
    train = train.append(cv)
    classifier = MultinomialNB()

    target = 'is_attributed'
    classifier.fit(train[predictors].values, train[target].values)

    output_dir = '../NB'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Make oofs ...")
    valid['pred'] = classifier.predict_proba(valid[predictors])[:, 1]
    valid.to_csv(output_dir+'/oof_pred.csv', index=False)

    print("Make predictions ...")
    test['pred'] = classifier.predict_proba(test[predictors])[:, 1]
    test.to_csv(output_dir+'/test_pred.csv', index=False)

    print("Done!!!")
    
