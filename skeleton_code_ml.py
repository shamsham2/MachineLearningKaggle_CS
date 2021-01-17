"""
This script can be used as skelton code to read the challenge train and test
csvs, to train a trivial model, and write data to the submission file.
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

## Read csvs

train_df = pd.read_csv('train_ml_4000.csv', index_col=0)
test_df = pd.read_csv('test_ml_1700.csv', index_col=0)

## Filtering column "mail_type"
train_x_cat = train_df[['mail_type','tld','org']]
train_x_num = train_df[['ccs','bcced','images','urls','salutations','designation','chars_in_subject','chars_in_body']]
train_x_cat = train_x_cat.fillna(value='None')
train_x_num = train_x_num.fillna(value=0.0)
train_y = train_df[['updates', 'personal', 'promotions',
                        'forums', 'purchases', 'travel',
                        'spam', 'social']]

test_x_cat = test_df[['mail_type','tld','org']]
test_x_num = test_df[['ccs','bcced','images','urls','salutations','designation','chars_in_subject','chars_in_body']]
test_x_cat = test_x_cat.fillna(value='None')
test_x_num = test_x_num.fillna(value=0.0)

## Do one hot encoding of categorical feature
feat_enc = OneHotEncoder()
feat_enc.fit(np.vstack([train_x_cat, test_x_cat]))
train_x_featurized = feat_enc.transform(train_x_cat)
test_x_featurized = feat_enc.transform(test_x_cat)

# combine featurized catigorial features and numerical features
train_x_combined = np.hstack([train_x_featurized.todense(),train_x_num.to_numpy()])
test_x_combined = np.hstack([test_x_featurized.todense(),test_x_num.to_numpy()])

## Train a simple OnveVsRestClassifier using featurized data
classif = OneVsRestClassifier(SVC(kernel='linear', probability=True))
classif.fit(train_x_combined, train_y)
pred_y = classif.predict_proba(test_x_combined)
print (pred_y.shape)

## Save results to submission file
pred_df = pd.DataFrame(pred_y, columns=['updates', 'personal', 'promotions',
                        'forums', 'purchases', 'travel',
                        'spam', 'social'])
pred_df.to_csv("knn_sample_submission_ml.csv", index=True, index_label='Id')
