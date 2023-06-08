import sys, gym, time
import highway_env
# active inference model setup
import os, time, argparse, pickle, cv2
from sys import argv
from distutils.dir_util import copy_tree
import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
import src.util as u
import src.tfutils as tfu
import src.tfloss as loss
from src.gym import *
from src.variables import *
from src.tfmodel import ActiveInferenceModel
import random
import tensorflow
from multiprocessing import Pool, TimeoutError
from graphs.reconstructions_plot import reconstructions_plot
from graphs.generate_traversals import generate_traversals
from graphs.stats_plot import stats_plot
from graphs.stats_test import stats_plot_test
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv

np_precision = np.float32
resume= True

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
parser = argparse.ArgumentParser(description='Training script.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if True: # If the machine used does not have enough memory, make this True
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
   
if __name__ == '__main__':
    
    start_time_5k=time.time()
    folder_chp = folder + '/checkpoints'
    try: os.mkdir(folder)
    except: print('Folder already exists!!')
    try: os.mkdir(folder_chp)
    except: print('Folder chp creation error')

    stats_start = {'test_total_reward':[],'Accuracy_X':[],'Accuracy_Y':[],'Accuracy_vx':[],'Accuracy_vy':[],'total_timesteps':[],'total_test_frames':[],'Actual_X':[],'Actual_Y':[],'Actual_vel_X':[],'Actual_vel_Y':[],'Rec_X':[],'Rec_Y':[],'Rec_vel_X':[],'Rec_vel_Y':[],
       'Pre_X':[],'Pre_Y':[],'Pre_vel_X':[],'Pre_vel_Y':[]}
       
    print("pi dim",pi_dim)

    t1 = ActiveInferenceModel(s_dim=s_dim,  pi_dim=pi_dim, gamma=gamma, beta_s=gamma_max, beta_o=beta_o)
    t5 = ActiveInferenceModel(s_dim=s_dim,  pi_dim=pi_dim, gamma=gamma, beta_s=gamma_max, beta_o=beta_o)
    t15= ActiveInferenceModel(s_dim=s_dim,  pi_dim=pi_dim, gamma=gamma, beta_s=gamma_max, beta_o=beta_o)

    t1.load_all("checkpoints_1")
    t5.load_all("checkpoints_5")
    t15.load_all("checkpoints_15")


    epoch = 0 # for now we count every frame as epoch (online learning) 
    #Initialising environment for testing
    env2 = DummyVecEnv([make_env(env_id, 0)]) # for plotting 
    #Getting observation for testing
    obs2=env2.reset()
    total_test_frames=0

    for game in range(100000):

        start_time=time.time()
        #obs=env.reset()
        global_temperature = 1.
        #model.steps=3
        t1.discount_factor = 0.99
        t1.term0_weight = 1. 
        t1.term1_weight = 1.
        t1.term2_weight = 1.
        t1.batch_size=1
        
        t5.discount_factor = 0.99
        t5.term0_weight = 1. 
        t5.term1_weight = 1.
        t5.term2_weight = 1.
        t5.batch_size=1

        t15.discount_factor = 0.99
        t15.term0_weight = 1. 
        t15.term1_weight = 1.
        t15.term2_weight = 1.
        t15.batch_size=1


        print("Testing ..........")

        _,obs2,env2,total_test_frames=rollout(epoch,t1,t5,t15,env2,obs2,stats_start,folder,folder_chp,no_of_actions,start_time_5k,total_test_frames,nr_episodes = 0,batch_size=1,test=True,render=False)
        if True: stats_plot_test(stats_start, folder+'/testing')










