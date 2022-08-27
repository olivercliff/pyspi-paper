import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from copy import copy

# Classifier stuff
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.use('TkAgg') # uncomment if you're having problems with multithreading

n_resamples = 30

datatype = 'BasicMotions'
nettype = 'discipline'

print('Loading data.')
with open('results/'+datatype+'-features.pkl','rb') as f:
    dat = cPickle.load(f)

X_train = dat['X_train']
y_train = dat['y_train']
X_test = dat['X_test']
y_test = dat['y_test']

spis = list(X_test.keys())
scores = pd.DataFrame(columns=[f'resample-{i}' for i in range(n_resamples)],
                        index=spis,
                        data=np.full((len(spis),n_resamples),np.nan))
scores.index.name = 'Statistic'

X_all = {}
for stat in X_train.keys():
    if stat not in spis:
        continue
    X_all[stat] = np.concatenate((X_train[stat],X_test[stat]))

def get_accuracy(X,y,n_train,n_test,scores):
    scaler = StandardScaler()
    for i in range(n_resamples):
        print(f'Resample {i}/{n_resamples}...')

        # Get the test/train split for labels
        y_train, y_test = train_test_split(y,train_size=n_train,test_size=n_test,random_state=i)

        for stat in X:

            X_train, X_test = train_test_split(X[stat],train_size=n_train,test_size=n_test,random_state=i)

            # Remove any features with 100% NaNs
            X_train = X_train[:,~np.isnan(X_train).any(axis=0)]
            X_test = X_test[:,~np.isnan(X_test).any(axis=0)]

            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

            try:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                clf.fit(X_train,y_train)
                scores.loc[stat].iloc[i] = balanced_accuracy_score(y_test,clf.predict(X_test))
            except (KeyError,ValueError) as e:
                print(f'Error for {stat} on resample {i}: {e}')

# Generate some shuffled data for nulls
y_train_null = shuffle(y_train)
y_test_null = shuffle(y_test)
nulls = copy(scores)

clf = SVC(kernel='linear')

get_accuracy(X_all,y_test+y_train,len(y_test),len(y_train),scores)
get_accuracy(X_all,y_test_null+y_train_null,len(y_test),len(y_train),nulls)

scores.to_csv('results/' + datatype + '-spis.csv')
nulls.to_csv('results/' + datatype + '-spis-null.csv')
plt.show()