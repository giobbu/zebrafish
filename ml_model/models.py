from tensorflow import keras
import tensorflow as tf
import numpy as np
from keras.utils.vis_utils import plot_model
from tensorflow.keras import regularizers

def LSTM(units_, units_fc, drop, inp_sqc, out_sqc, features, labels_xy, labels_vxy, lr, loss, l_1, l_2):            
    
    inp_ = tf.keras.layers.Input(shape = (inp_sqc, features))

    out = tf.keras.layers.LSTM(units_, return_sequences = True, kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2) )(inp_)
    out = tf.keras.layers.LSTM(units_, kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2) )(out)
    
    out = tf.keras.layers.RepeatVector(out_sqc)(out)
    out_1 = tf.keras.layers.LSTM(units_, return_sequences = True, kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2) )(out)
    out_2 = tf.keras.layers.LSTM(units_, return_sequences = True, kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 =
l_2) )(out)

    out_1 = tf.keras.layers.Dense(units_fc, activation='relu', kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2))(out_1)
    out_1 = tf.keras.layers.Dropout(drop)(out_1)
   
    out_2 = tf.keras.layers.Dense(units_fc, activation='relu', kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2))(out_2)
    out_2 = tf.keras.layers.Dropout(drop)(out_2)
    
    out_xy = tf.keras.layers.Dense(labels_xy)(out_1)
    out_vxy = tf.keras.layers.Dense(labels_vxy)(out_2)
          
    #  initialize model
    model = tf.keras.models.Model(inputs = inp_, outputs = [out_xy, out_vxy])
    opt = keras.optimizers.Adam(learning_rate = lr) 
    model.compile(optimizer = opt, loss = loss,  metrics = [keras.metrics.RootMeanSquaredError()])
    return model



def GRU(units_, units_fc, drop, inp_sqc, out_sqc, features, labels_xy, labels_vxy, lr, loss, l_1, l_2):            
    
    inp_ = tf.keras.layers.Input(shape = (inp_sqc, features))

    out = tf.keras.layers.GRU(units_, return_sequences = True, kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2) )(inp_)
    out = tf.keras.layers.GRU(units_, kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2) )(out)
    
    out = tf.keras.layers.RepeatVector(out_sqc)(out)
    out_1 = tf.keras.layers.GRU(units_, return_sequences = True, kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2) )(out)
    out_2 = tf.keras.layers.GRU(units_, return_sequences = True, kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 =
l_2) )(out)

    out_1 = tf.keras.layers.Dense(units_fc, activation='relu', kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2))(out_1)
    out_1 = tf.keras.layers.Dropout(drop)(out_1)
   
    out_2 = tf.keras.layers.Dense(units_fc, activation='relu', kernel_regularizer = regularizers.l1_l2(l1 = l_1, l2 = l_2))(out_2)
    out_2 = tf.keras.layers.Dropout(drop)(out_2)
    
    out_xy = tf.keras.layers.Dense(labels_xy)(out_1)
    out_vxy = tf.keras.layers.Dense(labels_vxy)(out_2)

#    out = tf.keras.layers.Dropout(drop)(out)    
#    out_ = tf.keras.layers.Reshape([out_sqc, labels])(out)
          
    #  initialize model
    model = tf.keras.models.Model(inputs = inp_, outputs = [out_xy, out_vxy])
    opt = keras.optimizers.Adam(learning_rate = lr) 
    model.compile(optimizer = opt, loss = loss,  metrics = [keras.metrics.RootMeanSquaredError()])
    return model





#model = GRU(100, 50,  0.1, 12, 4, 400, 2, 0.01, 'mse')
#keras.utils.plot_model(model, show_shapes=True, to_file='model.png')

