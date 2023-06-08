import numpy as np
import pickle
import tensorflow as tf
from src.variables import *

def convert_range(OldValue, OldMin, OldMax, NewMin, NewMax):
    return (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin

np_precision = np.float32

def softmax(x, temp):
    e_x = np.exp(x/temp)
    print("e_x"+str(e_x))
    return e_x/e_x.sum(axis=0)


def softmax_multi_with_log(x, single_values=5, eps=1e-20, temperature=1):
    """Compute softmax values for each sets of scores in x."""
    
    x = x.reshape(-1, single_values)
    #print("x"+str(x))
    #x=np.exp(x - np.max(x,1)).reshape(-1,1)
    #print("temperature in softmax",temperature)
    x = x - np.max(x,1).reshape(-1,1) # Normalization
    #print("x normalized"+str(x))
    e_x = np.exp(x/temperature)
    SM = e_x / e_x.sum(axis=1).reshape(-1,1)
    logSM = x - np.log(e_x.sum(axis=1).reshape(-1,1) + eps) # to avoid infs
    #print("--> softmax_multi_with_log")
    return SM, logSM

    

def encode(x):
    x=tf.reshape(x,[int(x.shape[0]/n_cars),int(x.shape[1]*n_cars)]) 
    return x

def decode(x):
    x=tf.reshape(x,[int(x.shape[0]*n_cars),int(x.shape[1]/n_cars)])  
    return x
    
"""
def compare_reward(po1, axis = [0], factor = 10):
    logpo1_reward = 0
    for i in range(n_cars):
        logpo1_reward += tf.square(po1[:,i*6-1]*factor - tf.ones_like(po1[:,i*6-1])*factor)   
    logpo1 = logpo1_reward / n_cars
    return -logpo1
"""
def compare_reward(po1, factor = 10.): #5.
    ''' Using MSE. '''

    logpo1_reward = tf.square(po1[:,-1]*factor - 1*factor)
    return -logpo1_reward 
