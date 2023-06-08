import sys, gym, time
import highway_env
# active inference model setup
from sklearn.metrics import r2_score,mean_squared_error
import os, time, argparse, pickle, cv2
from sys import argv
from distutils.dir_util import copy_tree
import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
import src.util as u
import src.tfutils as tfu
import src.tfloss as loss
from src.variables import *
from src.tfmodel import ActiveInferenceModel
import random
from multiprocessing import Pool, TimeoutError
from graphs.reconstructions_plot import reconstructions_plot
from graphs.generate_traversals import generate_traversals
from graphs.stats_plot import stats_plot
#from sklearn.metrics import mean_absolute_percentage_error
#from sklearn.metrics import median_absolute_error
from stable_baselines3.common.vec_env import SubprocVecEnv
np_precision = np.float32



def softmax(x, temp):
    e_x = np.exp(x/temp)
    return e_x/e_x.sum(axis=0)
    

'''
Top down precision
a: The sum a+d show the maximum value of omega
b: This shows the average value of D_kl[pi] that will cause half sigmoid (i.e. d+a/2)
c: This moves the steepness of the sigmoid
d: This is the minimum omega (when sigmoid is zero)
'''

def make_env(env_id, rank, seed=0):
    def _init():
        env=gym.make(env_id)

        config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "absolute": False, 
            "order": "sorted" }, 
        "policy_frequency": 15,
        "duration":600,
        "vehicles_count": 5,
        "simulation_frequency": 15,
        }

        env.configure(config)
        print("env_id", env_id)
        env.seed(seed + rank)
        return env
    return _init
    
def normalize_observation(obs_in, r_in, pi_dim = 5, batch_size = 16, tiles = 1):
    out = []
    out_repeated = []

    #print("Input to normalize_observation",obs_in)
    for o_single, r in zip(obs_in, r_in):
        o_single=o_single.flatten()
        if True:
            o_single = [max(el, -1) for el in o_single]
            o_single = [min(el, 1) for el in o_single]
            #r = max(r, -5)
            #r = min(r, 5)
            o_single = [u.convert_range(el, -1, 1, 0, 1) for el in o_single]
            #r = u.convert_range(r, -5, 5, 0, 1)
        else:
            o_single = [sigmoid(el) for el in o_single]
            #r = sigmoid(r) #*0.01
               
        o_single_r = np.append(np.tile(o_single, tiles), np.tile(r, tiles))
        
        out.append(o_single_r)
        out_repeated.append(np.tile(o_single_r, (pi_dim, 1)))
        
    out = np.float32(np.asarray(out))
    out_repeated = np.float32(np.asarray(out_repeated))
    out_repeated = np.reshape(out_repeated, [batch_size*pi_dim, -1])
    #print("obs_in after normalize_observation", out)
    return out, out_repeated

    
def future_timestep(env,pichoices,jumps,batch_size,render):
    rewards_first=0
    for i in range(jumps):
        if i==0:
           obs, rewards, dones, info = env.step(pichoices)
           if render: env.render()
           rewards_first=rewards[0]
        else:
           pichoices=np.full((batch_size),0)
           obs, rewards, dones, info = env.step(pichoices)
           rewards_first=rewards_first+rewards[0]
    return obs, rewards, dones, info,rewards_first

def calc_threshold(P, axis):
    return np.max(P,axis=axis) - np.mean(P,axis=axis)
    
def select_action(model,pi_one_hot,no_of_actions, o0,r, stats,test,steps=1,batch_size = 1, idle_policy = True):
    _, o_repeated = normalize_observation(o0, r, batch_size = batch_size)
    pi_one_hot=pi_one_hot.tolist()
    pi=np.array(pi_one_hot)
    pi=pi.astype('float32')
    pi_repeated=np.tile(pi,(batch_size, 1))
    

    sum_G, sum_terms, po2,steps_list = model.calculate_G_4_repeated_idle(o_repeated,pi_repeated,test,steps=steps, calc_mean=True,samples=samples)
    discount_normalization = 0.
    for t in range(steps): 
        discount_normalization += (model.discount_factor**t)
    
    print("discount_normalization",discount_normalization)
    G = sum_G.numpy() / discount_normalization
    term0 = -sum_terms[0].numpy() / discount_normalization
    term1 = sum_terms[1].numpy() / discount_normalization
    term2 = sum_terms[2].numpy() / discount_normalization
   

    Ppi, log_Ppi = u.softmax_multi_with_log(-G, no_of_actions) # Full active inference agent
    pi_choices = np.array([np.random.choice(no_of_actions,p=Ppi[i]) for i in range(batch_size)]) # The selected action

    # One hot version..
    pi0 = np.zeros((batch_size,no_of_actions), dtype=np_precision)
    pi0[np.arange(batch_size), pi_choices] = 1.0
    
    term0 = -sum_terms[0].numpy()
    term1 = sum_terms[1].numpy() 
    term2 = sum_terms[2].numpy()
    po2=np.split(po2,batch_size)
    po2=[value[pi_choices[i]] for i,value in enumerate(po2)]
    po2=np.array(po2)
    return pi0,Ppi, pi_choices, po2,G


def rollout(epoch,t1,t5,t15,env,observation,stats,folder,folder_chp,no_of_actions,start_time_5k,total_test_frames,nr_episodes = 10, batch_size=16,test=False,render=False):

    #skip = 0
    total_games=0
    total_reward = 0
    total_timesteps = 0
    r = [0 for i in range(batch_size)]
    r1 = [0 for i in range(batch_size)]
    obs=observation
    if render: env.render()
    done_counter=0
    
    while total_games <= nr_episodes:

            total_test_frames=total_test_frames+1
            print("total_test_frames",total_test_frames)
            o0_,r_ = obs,r1 # save copy for weights updates
            pi_one_hot = np.zeros((no_of_actions,no_of_actions))
            data = np.arange(no_of_actions)
            pi_one_hot[data, data] = 1
            
            #pi0, log_Ppi, pi_choices, po1,term0,term1,term2  = select_action(model,pi_one_hot,no_of_actions, obs,r,stats,test,steps=model.steps,batch_size=batch_size,idle_policy=True)
            pi0,_, _, po1,G_1= select_action(t1, pi_one_hot,no_of_actions, obs,r,stats,test,steps=1,batch_size=batch_size,idle_policy = False)#, steps=1) #t+1 by skipping 10 frames
            
            print("G_1",G_1)
            _,_, _, po1,G_5 = select_action(t5, pi_one_hot,no_of_actions, obs,r,stats,test,steps=5,batch_size=batch_size,idle_policy = True)#, steps=2) #t+1 by skipping 5 frames
            
            print("G_5",G_5)
            _,_, _, po1,G_15= select_action(t15, pi_one_hot,no_of_actions, obs,r,stats,test,steps=15,batch_size=batch_size,idle_policy = True)#,steps=10) #t+1 by skipping 1 frame
            
            print("G_15",G_15)

            G_all=G_1+G_5+G_15
            print("G_all",G_all)
            val, idx = min((val, idx) for (idx, val) in enumerate(G_all))
            """
            pi_choices=[idx]
            print("pi_choices",pi_choices)
            """

            Ppi, log_Ppi = u.softmax_multi_with_log(-G_all, no_of_actions) # Full active inference agent
            pi_choices = np.array([np.random.choice(no_of_actions,p=Ppi[i]) for i in range(batch_size)]) # The selected action


            obs, r1, dones, info,rewards_first = future_timestep(env,pi_choices,jumps,batch_size,render)
            #print("info",info)
            o1=obs
            o0 , _ = normalize_observation(o0_ , r_, batch_size = 1)
            o1 , _ = normalize_observation(o1, r1, batch_size = 1)

            total_reward += rewards_first
            total_timesteps=total_timesteps+1 
            
            qs0, qs0_mean,_ = t15.model_down.encoder_with_sample(o0)
            ps1, ps1_mean, ps1_logvar = t15.model_mid.transition_with_sample(pi0, qs0_mean)
            po1 = t15.model_down.decoder(ps1_mean)
            
            qs1_mean, qs1_logvar = t15.model_down.encoder(o1)
            po1_rec =t15.model_down.decoder(qs1_mean)
            

            x_predicted=po1[0][6]
            x_actual=o1[0][6]
            y_predicted=po1[0][7]
            y_actual=o1[0][7]
            vx_predicted=po1[0][8]
            vx_actual=o1[0][8]
            vy_predicted=po1[0][9]
            vy_actual=o1[0][9]
            abs_x=abs(x_predicted - x_actual)/x_actual
            Accuracy_X  =  (1-abs_x) * 100
            abs_y       =   abs(y_predicted - y_actual)/y_actual
            Accuracy_Y  =  (1-abs_y) * 100
            abs_vx      =   abs(vx_predicted - vx_actual)/vx_actual

            Accuracy_vx =  (1-abs_vx) * 100
            abs_vy      =   abs(vy_predicted - vy_actual)/vy_actual
            Accuracy_vy =  (1-abs_vy) * 100
 
            stats['Accuracy_X'].append(Accuracy_X)
            stats['Accuracy_Y'].append(Accuracy_Y)
            stats['Accuracy_vx'].append(Accuracy_vx)
            stats['Accuracy_vy'].append(Accuracy_vy)
            stats['total_test_frames'].append(total_test_frames)
            print("Prediction",po1[0][6:10])
            print("Actual",o1[0][6:10])
            print("Accuracy_X",Accuracy_X)
            print("Accuracy_Y",Accuracy_Y)
            print("Accuracy_vx",Accuracy_vx)
            print("Accuracy_vy",Accuracy_vy)
            mse_po1=tfu.mean_squared_error(po1[0],o1[0])
         
            if dones[0]: # log total results for first env in batch
                if render: env.render()
                total_games += 1
                if test: stats['test_total_reward'].append(total_reward)
                else: stats['total_reward'].append(total_reward) 
                print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))     
                stats['total_timesteps'].append(total_timesteps)      
                total_timesteps=0
                total_reward=0     

            if not test:
                       model, epoch ,mse_reward_o1= train_single_step(model, o0, o1, pi0, log_Ppi, pi_choices,stats,folder,folder_chp,optimizers,start_time_5k,epoch)
                    
                       
                       qs0, qs0_mean,_ = model.model_down.encoder_with_sample(o0)
                       ps1, ps1_mean, ps1_logvar = model.model_mid.transition_with_sample(pi0, qs0_mean)
                       po1 = model.model_down.decoder(ps1_mean)

                       stats['Actual_X_o1'].append(o1[0][6])
                       stats['Actual_Y_o1'].append(o1[0][7])
                       stats['Actual_vel_X_o1'].append(o1[0][8])
                       stats['Actual_vel_Y_o1'].append(o1[0][9])
                       stats['mse_po1'].append(mse_po1)
                       stats['Pre_X'].append(po1[0][6])
                       stats['Pre_Y'].append(po1[0][7])
                       stats['Pre_vel_X'].append(po1[0][8])
                       stats['Pre_vel_Y'].append(po1[0][9])
                       stats["predicted_reward_mid"].append(po1[0][-1])
                       predicted_reward_mid_error=tfu.mean_squared_error(po1[0][-1],o1[0][-1])
                       stats["predicted_reward_mid_error"].append(predicted_reward_mid_error)
                       stats['term0'].append(np.max(term0)-np.min(term0))
                       stats['term1'].append(np.max(term1)-np.min(term1))
                       stats['term2'].append(np.max(term2)-np.min(term2))
                       if epoch % 1000== 0: 
                                           with open('test_stats/stats_train.pkl','wb') as ff:
                                               pickle.dump(stats,ff)  
                                               
            if test:
                    stats['Actual_X'].append(o1[0][6])
                    stats['Actual_Y'].append(o1[0][7])
                    stats['Actual_vel_X'].append(o1[0][8])
                    stats['Actual_vel_Y'].append(o1[0][9])
                    stats['Pre_X'].append(po1[0][6])
                    stats['Pre_Y'].append(po1[0][7])
                    stats['Pre_vel_X'].append(po1[0][8])
                    stats['Pre_vel_Y'].append(po1[0][9]) 
                    stats['Rec_X'].append(po1_rec[0][6])
                    stats['Rec_Y'].append(po1_rec[0][7])
                    stats['Rec_vel_X'].append(po1_rec[0][8])
                    stats['Rec_vel_Y'].append(po1_rec[0][9])

                    if total_test_frames % 1000== 0: 
                                                    with open('test_stats/stats_test.pkl','wb') as ff:
                                                        pickle.dump(stats,ff)          
    return epoch,observation,env,total_test_frames