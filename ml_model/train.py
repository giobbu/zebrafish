import numpy as np
import argparse
import pandas as pd
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import gc
import pickle
import tensorflow as tf
from ml_utils import tf_data_loader, set_seed, inverse_transform, fish_movie_pred_targ
from models import LSTM
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from tensorflow import keras
import matplotlib.animation as animation
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--i', default = 12, type=int, help = 'Input sequence')
parser.add_argument('--o', default = 4, type=int, help = 'Output sequence')
parser.add_argument('--b', default = 32, type=int, help = 'Batch size')
parser.add_argument('--s', default = 42, type=int, help = 'set seed')
parser.add_argument('--ut', default = 100, type=int, help = 'set units temporal layer')
parser.add_argument('--ufc', default = 50, type=int, help = 'set units fully connected')
parser.add_argument('--lr', default = 0.01, type= float, help = 'set learning rate')
parser.add_argument('--dr', default = 0.0, type= float, help = 'set drop rate')
parser.add_argument('--ep', default = 10, type=int, help = 'set number epochs')
parser.add_argument('--l1', default = 0.0, type=float, help = 'set reg l1')
parser.add_argument('--l2', default = 0.0, type=float, help = 'set reg l2')
parser.add_argument('--save', default = 'model_trial', type=str, help = 'save trained model')
args = parser.parse_args()

inp_sqc = args.i
out_sqc = args.o
batch_sz = args.b
drop = args.dr
lr = args.lr
units_ = args.ut
units_fc = args.ufc
seed = args.s
n_epochs = args.ep
l1 = args.l1
l2 = args.l2
filename = args.save
 
# load scaled train, val, test sets
with open(r"scaled_train_val_test.pkl", "rb") as input_file:
     scaler, sc_train, sc_val, sc_test = pickle.load(input_file)

# reproducibility
set_seed(seed)

# prepare data tf
tf_train = tf_data_loader(sc_train, inp_sqc, out_sqc, batch_sz)
tf_val = tf_data_loader(sc_val, inp_sqc, out_sqc, batch_sz)
tf_test = tf_data_loader(sc_test, inp_sqc, out_sqc, 1)

# upload model
features = sc_train.shape[1]
labels_xy = 2  # coordinates x-y
labels_vxy = 2  # coordinates vx-vy
model = LSTM(units_, units_fc , drop, inp_sqc, out_sqc, features, labels_xy, labels_vxy, lr, 'mse', l1, l2)
keras.utils.plot_model(model, show_shapes=True, to_file='model.png')

# set early stop to avoid overfitting
#early_stopping = keras.callbacks.EarlyStopping(
#            patience = GCF.EARLY_STOPPING_PATIENCE,
#            min_delta = GCF.EARLY_STOPPING_MIN_DELTA,
#            restore_best_weights = True)

# train model
gpu_strategy = tf.distribute.get_strategy()
with gpu_strategy.scope():
     history = model.fit(tf_train, validation_data = tf_val, epochs = n_epochs)

# plot history training and validation loss
pd.DataFrame(history.history, columns = ["loss", "val_loss"] ).plot()
plt.title("mse")
plt.show()

# save the model
model.save(filename)

