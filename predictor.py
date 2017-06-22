# import XGBoost (using Anaconda)
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']


# import relevant libraries
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


# ==== PREPROCESSING ==== 
train = pd.read_csv("training-data.tsv", header=0, delimiter="\t")
train['total_price'] = train['price']
train.loc[train.shipping_cost != -1, 'total_price'] += train.shipping_cost

sold_items = train[train.sales != 0]
train.loc[train.sales != 0, 'clicks'] = train.clicks / train.sales
train.loc[train.sales != 0, 'sales'] = 1


# CountVectorizer can also be used
vectorizer = TfidfVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None, \
                             token_pattern = r'[-]?[0-9]+.') # or r'[-]?[0-9]*\.?[0-9]+.' #
names = vectorizer.fit_transform(train['name'])
des = vectorizer.fit_transform(train['description'].values.astype('str'))


# ==== LEARNING & EVALUATION ==== 
y = train.sales
X = sp.sparse.hstack((train[['total_price', 'clicks']], train.loc[:,'merchant_id':'sku'], names, des)).tocsr()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
	
		
# XGBoost
xgc = XGBClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3)
xgc.fit(X_train, y_train)

p_pred_xgc = xgc.predict_proba(X_test)
print 'Probabilities'
print p_pred_xgc[:,1]

# Evaluation
fpr, tpr, thresholds = roc_curve(y_test, p_pred_xgc[:,1], pos_label=1)
roc_auc_xgc = auc(fpr, tpr)
print 'AUC' 
print roc_auc_xgc

plt.figure()
lw = 1
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_xgc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
