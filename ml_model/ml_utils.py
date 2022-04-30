import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import gc
import pickle
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Reproducability
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'    
 
   
# invert data to the original scale
def inverse_transform(prediction, scaler):
    # invert scaling
    inv_pred = scaler.inverse_transform(prediction)
    return inv_pred


# transform data for dl model prediction
def tf_data_loader(dataset, inp_sqc, out_sqc, batch_sz):
    # features and labels
    features = tf.data.Dataset.from_tensor_slices(dataset)    
    labels_xy = tf.data.Dataset.from_tensor_slices(dataset[:, :2])
    labels_vxy = tf.data.Dataset.from_tensor_slices(dataset[:, 2:4])
    # features - past observations
    past_obs = features.window(inp_sqc,  shift=1,  stride=1,  drop_remainder=True) 
    past_obs = past_obs.flat_map(lambda window: window.batch(inp_sqc))
    # labels - future observations
    # position
    fut_obs_xy = labels_xy.window(out_sqc, shift=1,  stride=1,  drop_remainder=True).skip(inp_sqc)
    fut_obs_xy = fut_obs_xy.flat_map(lambda window: window.batch(out_sqc))
    # velocity
    fut_obs_vxy = labels_vxy.window(out_sqc, shift=1,  stride=1,  drop_remainder=True).skip(inp_sqc)
    fut_obs_vxy = fut_obs_vxy.flat_map(lambda window: window.batch(out_sqc))
    # create dataset feat (past) and labels (future)   
    tf_dataset = tf.data.Dataset.zip((past_obs, (fut_obs_xy, fut_obs_vxy)))
    # divide in batches
    tf_dataset = tf_dataset.batch(batch_sz).prefetch(tf.data.experimental.AUTOTUNE)
    return tf_dataset






# plot experiment movie of zebrafish cosine
def fish_movie_pred_targ(traj_pred, traj_targ, n_frames, title, interv):
   
    # First set up the frame coordinates
    # predictions
    frame_pred = traj_pred[0]
    x_pred, y_pred = frame_pred[:, 0], frame_pred[:, 1]
    vx_pred, vy_pred = frame_pred[:, 2], frame_pred[:, 3]
    # targets
    frame_targ = traj_targ[0]
    x_targ, y_targ = frame_targ[:, 0], frame_targ[:, 1]
    vx_targ, vy_targ = frame_targ[:, 2], frame_targ[:, 3]

    def animate(i): 
        # predictions
        frame_pred = traj_pred[i]
        x_pred, y_pred = frame_pred[:, 0], frame_pred[:, 1]
        vx_pred, vy_pred = frame_pred[:, 2], frame_pred[:, 3]    
        # targets
        frame_targ = traj_targ[i]
        x_targ, y_targ = frame_targ[:, 0], frame_targ[:, 1]
        vx_targ, vy_targ = frame_targ[:, 2], frame_targ[:, 3]

        tt.set_text('Frame: ' + str(i))
        # predictions scatter and quiver
        sc_pred.set_offsets(frame_pred)
        arr_pred.set_offsets(frame_pred)
        arr_pred.set_UVC(vx_pred, vy_pred)
        # targets scatter and quiver
        sc_targ.set_offsets(frame_targ)
        arr_targ.set_offsets(frame_targ)
        arr_targ.set_UVC(vx_targ, vy_targ)
        return sc_pred, arr_pred, sc_targ, arr_targ
    
    fig, ax = plt.subplots(figsize=(5,5)) 
    sc_pred = ax.scatter(x_pred, y_pred, s=2, color ='red')
    arr_pred = ax.quiver(x_pred, y_pred, vx_pred, vy_pred, color='orange')
    sc_targ = ax.scatter(x_targ, y_targ, s=2, color ='green')
    arr_targ = ax.quiver(x_targ, y_targ, vx_targ, vy_targ, color='blue')
    tt = ax.text(0., 1.05, 'Frame: ', transform = ax.transAxes)

    # call the animator
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interv)
    return anim









