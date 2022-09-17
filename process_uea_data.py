import numpy as np
import _pickle as cPickle

import tslearn.datasets

try:
    from pyspi.calculator import CalculatorFrame
except ModuleNotFoundError:
    # You're probably using the pynats branch
    from pynats.calculator import CalculatorFrame

uea_ds = 'BasicMotions'

data_loader = tslearn.datasets.UCR_UEA_datasets()

X_train, y_train, X_test, y_test = data_loader.load_dataset(uea_ds)

names_train = [uea_ds+f'-{i}-{y}' for i, y in enumerate(y_train)]
names_test = [uea_ds+f'-{i}-{y}' for i, y in enumerate(y_test)]

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

# Combine the frames

# Assign numeric value to classes
labels = list(np.unique(y_train))

cf_train.set_group(labels)
cf_test.set_group(labels)

# This will take a while
cf_train.compute()
cf_test.compute()

# Create a data matrix from the results
features_train = {}
for calc in cf_train.calculators.values:
    for spi in calc[0].spis.keys():
        dat = np.atleast_2d(calc[0].table[spi].values.ravel())
        try:
            features_train[spi] = np.concatenate((dat,features_train[spi]),axis=0)
        except:
            features_train[spi] = dat

features_test = {}
for calc in cf_test.calculators.values:
    for spi in calc[0].spis.keys():
        dat = np.atleast_2d(calc[0].table[spi].values.ravel())
        try:
            features_test[spi] = np.concatenate((dat,features_test[spi]),axis=0)
        except:
            features_test[spi] = dat

with open(f'results/{uea_ds}-features.pkl','wb') as f:
    cPickle.dump(dict(X_train=features_train, X_test=features_test,
                        y_train=cf_train.groups, y_test=cf_test.groups), f)