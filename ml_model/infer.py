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
parser.add_argument('--i', type=int, help = 'input sequence (see file trained model)')
parser.add_argument('--o', type=int, help = 'output sequence (see file trained model)')
parser.add_argument('--it', default = False, type=bool, help = 'iteratively visualize model prediction')
parser.add_argument('--n', default = 'model_trial', type=str, help = 'trained model file name')
args = parser.parse_args()

iterative = args.it
filename = args.n
inp_sqc = args.i
out_sqc = args.o

# load trained model
model = keras.models.load_model(filename)

# load scaled train, val, test sets
with open(r"scaled_train_val_test.pkl", "rb") as input_file:
     scaler, _, _, sc_test = pickle.load(input_file)

# prepare data tf
tf_test = tf_data_loader(sc_test, inp_sqc, out_sqc, 1)

# start loop inference
traj_pred_l, traj_targ_l = [], []
for (step, (inp_, targ_)) in tqdm(enumerate(tf_test)):

    # get predictions
    pred_xy, pred_vxy = model.predict(inp_)
    xy, vxy = targ_[0], targ_[1]

    #xy = inverse_transform(pred, scaler) 
    #t = inverse_transform(targ_[0], scaler)

    if iterative == True: 
       # plot positions focal fish
       plt.scatter(pred[0, 0], pred[0, 1], color='orange' )
       plt.scatter(pred[1:, 0], pred[1:, 1], color='yellow' )
       plt.scatter(targ_[0][:, 0], targ_[0][:, 1] )
       plt.xlim(-1, 1)
       plt.ylim(-1, 1)
    #plt.ylim(0, 3500)
       plt.show()

    traj_pred_l.append(np.hstack([pred_xy[0], pred_vxy[0]]))
    traj_targ_l.append(np.hstack([xy[0], vxy[0]])) 

    
#    if step == 100:
#       break
    # plot velocity focal fish
#    fig, ax = plt.subplots(figsize=(5,5))
#    ax.scatter(pred_xy[0][:, 0], pred_xy[0][:, 1], color='yellow' )
#    ax.quiver(pred_xy[0][:, 0],  pred_xy[0][:, 1],  pred_vxy[0][:, 0],  pred_vxy[0][:, 1], color = 'orange')
#    ax.scatter(xy[0][:, 0], xy[0][:, 1], color='blue')
#    plt.quiver(xy[0][:, 0], xy[0][:, 1], vxy[0][:, 0], vxy[0][:, 1], color='green')
#    plt.xlim(-1, 1)
#    plt.ylim(-1, 1)
#    plt.show()

# animated plot comparing predictions and targets
anim = fish_movie_pred_targ(traj_pred_l, traj_targ_l, 3000, 'trial', 10)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

writergif = animation.PillowWriter(fps=30)
anim.save('lines.gif', writer = writergif)



