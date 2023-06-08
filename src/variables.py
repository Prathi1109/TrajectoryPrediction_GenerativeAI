import tensorflow as tf 

s_dim=32
pi_dim=5

#Global variables

var_a = 0.0001;         
var_b = 1000.0;          
var_c = 0.0001;         
var_d = 0.0001      

''' KL / reconstruction loss weighting '''
beta_s = 10;    
beta_s_naive=10; 
beta_o = 1.0; 

''' KL annealing'''
gamma = 0.0;         
gamma_rate = 0.00001;     
gamma_max = 1.0;     
gamma_delay = 0 

''' Learning rates'''
l_rate_mid = 0.001;    l_rate_down = 0.0005

save_interval = 5000
plot_interval = 1
plot_stats_interval=500
reconstruct_interval=1
gamma_rate = 0.00001; 
gamma_delay = 0 
var_a = 0.0001; 
batch=16

no_of_actions=5
n_cars=5

var_a = 0.0001;         
var_b = 1000.0;          
var_c = 0.0001;         
var_d = 0.0001      

''' Training'''
ROUNDS = 1;       
TEST_SIZE = 1;      
epochs = 100000
samples = 10
repeats = 1
deepness = 1;  

        
stepsize=1
pi_dim = no_of_actions
env_id ='highway-v0'


#steps=1
jumps=1
calc_mean_global = True
tf_precision = tf.float32
global_temperature =  1

# Create folder to save model
signature = 'final_model_'
signature += str(gamma_rate)+'_'+str(gamma_delay)+'_'+str(var_a)
folder ='figs_'+signature
