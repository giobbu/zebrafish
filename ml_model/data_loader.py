import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gc
import pickle
import tensorflow as tf
from ml_utils import tf_data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--tr', default = 80, type=int, help = 'Percentage training set')
parser.add_argument('--val', default = 10, type=int, help = 'Percentage validation set')
parser.add_argument('--fld', default = 5, type=int, help = 'Number of folders for cross-validation')
args = parser.parse_args()

tr_perc = args.tr
val_perc = args.val
split = args.fld

# load dataset 
dataset = np.load('dataset.npy', allow_pickle=True)

# split in training and testing sets
perc = int(len(dataset)*(tr_perc/100))
train, test = dataset[:perc, :], dataset[perc:, :]

# standardize features
scaler = MinMaxScaler(feature_range=(-1, 1))

# split in folds for cross-validation
tscv = TimeSeriesSplit(n_splits = split)
for fold, (tr_index, val_index) in enumerate(tscv.split(train)):

    # get folder for train and validation
    fold_tr, fold_val = train[tr_index, :], train[val_index,:]
    
    # scale the folder data
    sc_fold_tr = scaler.fit_transform(fold_tr)
    sc_fold_val = scaler.transform(fold_val)

    # save in pickle format
    pickle.dump([scaler, sc_fold_tr, sc_fold_val] , open('fold_'+str(fold+1)+'.pkl', "wb"))
    
    del fold_tr, fold_val, sc_fold_tr, sc_fold_val
    gc.collect()

# scale train and test sets for final model evaluation
sc_train = scaler.fit_transform(train)
sc_test = scaler.transform(test)

# save scaled train and test to pickle
pickle.dump([scaler, sc_train, sc_test] , open('scaled_train_test.pkl', "wb"))

# split in train, vald and testing sets
perc = int(len(train)*(val_perc/100))
val_, train_ = train[:perc, :], train[perc:, :]
scaler = MinMaxScaler(feature_range=(-1, 1))
sc_train = scaler.fit_transform(train_)
sc_val = scaler.transform(val_)
sc_test = scaler.transform(test)
pickle.dump([scaler, sc_train, sc_val, sc_test] , open('scaled_train_val_test.pkl', "wb"))

print(dataset.shape)
print(train_.shape)
print(val_.shape)
print(test.shape)
print('')
print('X-Y')
print(train_[:2, :2])
print(sc_train[:2, :2])
print('')
print(train_[2:4, 2:4])
print(sc_train[2:4, 2:4])
########### try tf-data-loader
#batch_size = 32
#sample = tf_data_loader(sc_train, 12, 4, 32)
#print(next(iter(sample))[0].shape)
#print(next(iter(sample))[1].shape)
#print(train[:5, :4])







