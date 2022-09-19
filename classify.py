import _pickle as cPickle
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
n_sequential_features = 0

datatype = 'BasicMotions'

print('Loading data.')
with open('results/' + datatype + '-features.pkl','rb') as f:
    dat = cPickle.load(f)

X_train = dat['X_train']
y_train = dat['y_train']
X_test = dat['X_test']
y_test = dat['y_test']

classes = ['badminton','standing','running','walking']

# A bunch of SPIs were not included in the final analysis but are still in the toolkit
spis = pd.read_csv('data/spis.csv', index_col=0).index
scores = pd.DataFrame(columns=[f'resample-{i}' for i in range(n_resamples)],
                        index=spis,
                        data=np.full((len(spis), n_resamples), np.nan))
scores.index.name = 'Statistic'

def get_accuracy(X, y, n_train, n_test, scores):
    scaler = StandardScaler()
    for i in range(n_resamples):
        print(f'Resample {i}/{n_resamples}...')

        # Get the test/train split for labels
        y_train, y_test = train_test_split(y, train_size=n_train, test_size=n_test, random_state=i)

        for stat in X:

            X_train, X_test = train_test_split(X[stat], train_size=n_train, test_size=n_test, random_state=i)

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

X = {}
for spi in spis:
    X[spi] = np.concatenate((X_train[spi], X_test[spi]))

get_accuracy(X, y_test + y_train, len(y_test), len(y_train), scores)
get_accuracy(X, y_test_null + y_train_null, len(y_test), len(y_train), nulls)

scores.to_csv('results/spis.csv')
nulls.to_csv('results/null.csv')

print(f'Computing accuracy for all statistics...')
for stat in spis:
    try:
        X_train_combined = np.concatenate([X_train_combined, X_train[stat]],axis=1)
        X_test_combined = np.concatenate([X_test_combined, X_test[stat]],axis=1)
    except NameError:
        X_train_combined = X_train[stat]
        X_test_combined = X_test[stat]

X_combined = np.concatenate((X_train_combined, X_test_combined))
y_combined = np.concatenate((y_train, y_test))

dropids = np.where((~np.isfinite(X_combined)).all(axis=0))[0]

X_combined = np.delete(X_combined,dropids,axis=1)
X_train_combined = np.delete(X_train_combined,dropids,axis=1)
X_test_combined = np.delete(X_test_combined,dropids,axis=1)

clf = SVC(kernel='linear')
scores = pd.Series(index=range(n_resamples))
scores.name = 'Score'

for i in range(n_resamples):
    print(f'Resample {i}/{n_resamples}')
    if i == 0:
        X_train, X_test, y_train, y_test = X_train_combined, X_test_combined, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_combined,y_combined,
                                                            test_size=len(y_test),train_size=len(y_train),
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

scores.to_csv('results/all.csv')

print(f'Balanced accuracy for all features: {scores.mean()}')