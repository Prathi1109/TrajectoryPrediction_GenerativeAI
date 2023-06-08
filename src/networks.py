import tensorflow as tf
import pickle
import numpy as np
from shutil import copyfile
from src.tfutils import *

hidden_act = tf.nn.relu
dec_out_act = tf.nn.sigmoid
tiles = 1
    
def qs_net(self, opts):
    s_dim, pi_dim, tf_precision, precision = opts
    qs_net =  tf.keras.Sequential([
                  tf.keras.layers.InputLayer(input_shape=(26*tiles,)),
                  tf.keras.layers.Flatten(),
                  tf.keras.layers.Dense(int(256), activation=hidden_act, kernel_initializer='he_uniform'),
                  tf.keras.layers.Dropout(0.5),
                  tf.keras.layers.Dense(int(256), activation=hidden_act, kernel_initializer='he_uniform'),
                  tf.keras.layers.Dropout(0.5),
                  tf.keras.layers.Dense(s_dim + s_dim) #, activation = tf.nn.sigmoid activation = None activation = None
                  ]) # No activation
    return qs_net

def po_net_dense(self, opts):
    s_dim, pi_dim, tf_precision, precision = opts
    po_net_dense = tf.keras.Sequential([
                  tf.keras.layers.InputLayer(input_shape=(s_dim,)),
                  tf.keras.layers.Dense(int(256), activation=hidden_act, kernel_initializer='he_uniform'),
                  tf.keras.layers.Dropout(0.5),
                  tf.keras.layers.Dense(int(256), activation=hidden_act, kernel_initializer='he_uniform'),
                  tf.keras.layers.Dropout(0.5),
                  tf.keras.layers.Dense(26*tiles, kernel_initializer='he_uniform'), #activation=None, 
                  tf.keras.layers.Reshape(target_shape=(26*tiles,))
                  ])
    return po_net_dense

