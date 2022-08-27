import _pickle as cPickle
import numpy as np
import pandas as pd

# Classifier stuff
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.use('TkAgg') # uncomment if you're having problems with multithreading

n_resamples = 30
n_sequential_features = 0

datatype = 'BasicMotions'
nettype = 'discipline'

print('Loading data.')
with open('results/' + datatype + '-features.pkl','rb') as f:
    dat = cPickle.load(f)

X_train_stat = dat['X_train']
y_train_orig = dat['y_train']
X_test_stat = dat['X_test']
y_test_orig = dat['y_test']

classes = ['badminton','standing','running','walking']

spis = list(X_train_stat.keys())

print(f'Computing accuracy for all statistics...')
for stat in spis:
    try:
        X_train_all = np.concatenate([X_train_all,X_train_stat[stat]],axis=1)
        X_test_all = np.concatenate([X_test_all,X_test_stat[stat]],axis=1)
    except NameError:
        X_train_all = X_train_stat[stat]
        X_test_all = X_test_stat[stat]

X_all = np.concatenate((X_train_all,X_test_all))
y_all = np.concatenate((y_train_orig,y_test_orig))

dropids = np.where((~np.isfinite(X_all)).all(axis=0))[0]

X_all = np.delete(X_all,dropids,axis=1)
X_train_all = np.delete(X_train_all,dropids,axis=1)
X_test_all = np.delete(X_test_all,dropids,axis=1)

clf = SVC(kernel='linear')
scores = pd.Series(index=range(n_resamples))
scores.name = 'Score'

for i in range(n_resamples):
    print(f'Resample {i}/{n_resamples}')
    if i == 0:
        X_train, X_test, y_train, y_test = X_train_all, X_test_all, y_train_orig, y_test_orig
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_all,y_all,
                                                            test_size=len(y_test_orig),train_size=len(y_train_orig),
                                                            random_state=i)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    dropids = np.union1d(np.where((~np.isfinite(X_train)).any(axis=0))[0],
                                np.where((~np.isfinite(X_test)).any(axis=0))[0])

    X_train = np.delete(X_train,dropids,axis=1)
    X_test = np.delete(X_test,dropids,axis=1)

    clf.fit(X_train,y_train)
    score = balanced_accuracy_score(y_test,clf.predict(X_test))
    scores[i] = score
    print(f'Accuracy: {score}')

scores.to_csv('results/'+datatype+'-all.csv')

print(f'Balanced accuracy for all features: {scores.mean()}')