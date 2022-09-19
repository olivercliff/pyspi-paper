import numpy as np
import _pickle as cPickle

import tslearn.datasets

try:
    from pyspi.calculator import CalculatorFrame
except ModuleNotFoundError:
    print("Using legacy code.")
    from pynats.calculator import CalculatorFrame

dataset = 'BasicMotions'

data_loader = tslearn.datasets.UCR_UEA_datasets()

X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)

names_train = [f'{dataset}-{i}-{y}' for i, y in enumerate(y_train)]
names_test = [f'{dataset}-{i}-{y}' for i, y in enumerate(y_test)]

X_train = np.transpose(X_train, axes=(0,2,1))
X_test = np.transpose(X_test, axes=(0,2,1))

try:
    cf_train = CalculatorFrame(datasets=X_train, labels=y_train[:, np.newaxis],
                                names=names_train, fast=True)
    cf_test = CalculatorFrame(datasets=X_test, labels=y_train[:, np.newaxis],
                                names=names_test, fast=True)
except TypeError:
    cf_train = CalculatorFrame(datasets=X_train, labels=y_train[:, np.newaxis], names=names_train)
    cf_test = CalculatorFrame(datasets=X_test, labels=y_train[:, np.newaxis], names=names_test)

# Assign numeric value to classes
labels = list(np.unique(y_train))

cf_train.set_group(labels)
cf_test.set_group(labels)

# This will take a while
cf_train.compute()
cf_test.compute()

def extract_features(cf):
    features = {}
    try:
        for calc in cf.calculators.values:
            for spi in calc[0].spis.keys():
                dat = np.atleast_2d(calc[0].table[spi].values.ravel())
                try:
                    features[spi] = np.concatenate((dat, features[spi]), axis=0)
                except:
                    features[spi] = dat
    except AttributeError:
        for calc in cf.calculators.values:
            for s, spi in enumerate(calc[0]._statnames):
                dat = np.atleast_2d(calc[0].adjacency[s].ravel())
                try:
                    features[spi] = np.concatenate((dat, features[spi]), axis=0)
                except:
                    features[spi] = dat
    return features

# Create a data matrix from the results
features_train = extract_features(cf_train)
features_test = extract_features(cf_test)

with open(f'results/{dataset}-features.pkl','wb') as f:
    cPickle.dump(dict(X_train=features_train, X_test=features_test,
                        y_train=cf_train.groups, y_test=cf_test.groups), f)