import tensorflow as tf
import pickle
import numpy as np
from shutil import copyfile
from src.tfutils import *
import src.networks as networks
from src.networks import *
import src.util as u
from src.variables import *
import tensorflow_probability as tfp


class ModelMid(tf.keras.Model):
    def __init__(self, s_dim, pi_dim, tf_precision, precision):
        super(ModelMid, self).__init__()

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)

        self.s_dim =  32
        self.pi_dim = pi_dim
        self.ps_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(self.pi_dim+self.s_dim)),
              tf.keras.layers.Dense(units=1024, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(units=1024, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(units=1024, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(self.s_dim + self.s_dim),]) # No activation
        
    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def transition(self, pi, s0):
        mean, logvar = tf.split(self.ps_net(tf.concat([pi,s0],1)), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def transition_with_sample(self, pi, s0):
        ps1_mean, ps1_logvar = self.transition(pi, s0)
        ps1 = self.reparameterize(ps1_mean, ps1_logvar)
        return ps1, ps1_mean, ps1_logvar
    

class ModelDown(tf.keras.Model):
    def __init__(self, s_dim, pi_dim, tf_precision, precision):
        super(ModelDown, self).__init__()

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)
        self.s_dim = s_dim
        self.pi_dim = pi_dim
        

        opts = [s_dim, pi_dim, tf_precision, precision]
        
        self.qs_net = networks.qs_net(self, opts) 
        self.po_net = networks.po_net_dense(self, opts) # dense observation decoder
        
    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def encoder(self, o):
        mean_s, logvar_s = tf.split(self.qs_net(o), num_or_size_splits=2, axis=1)
        return mean_s, logvar_s
    """
    @tf.function
    def decoder(self, s):
        po_vehicles_batch = self.po_net(s)
        po_stacked=u.encode(po_vehicles_batch)
        return po_stacked
    """
    @tf.function
    def decoder(self, s):
        po = self.po_net(s)
        return po
    
    @tf.function
    def encoder_with_sample(self, o):
        mean, logvar = self.encoder(o)
        s = self.reparameterize(mean, logvar)
        return s, mean, logvar


class ActiveInferenceModel:
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o,habitual_planning = False, longterm_planning = False, habit_actions = False ,steps = 1,jumps = 1, discount_factor = 1., 
                 term0_weight = 1000., term1_weight = 10., term2_weight = 1000.,batch_size=1):
        
        self.habitual_planning = False
        self.longterm_planning = False
        self.habit_actions = habit_actions
        self.tf_precision = tf.float32
        self.precision = 'float32'
        self.steps = steps
        self.jumps = jumps
        self.discount_factor = discount_factor
        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.batch_size = batch_size
        tf.keras.backend.set_floatx(self.precision)
        self.term0_weight = term0_weight #1000.
        self.term1_weight = term1_weight #10.
        self.term2_weight = term2_weight #1000.
        
        if self.pi_dim > 0:
            #self.model_top = ModelTop(s_dim, pi_dim, self.tf_precision, self.precision)
            self.model_mid = ModelMid(s_dim, pi_dim, self.tf_precision, self.precision)
        self.model_down = ModelDown(s_dim, pi_dim, self.tf_precision, self.precision)

        self.model_down.beta_s = tf.Variable(beta_s, trainable=False, name="beta_s")
        self.model_down.gamma = tf.Variable(gamma, trainable=False, name="gamma")
        self.model_down.beta_o = tf.Variable(beta_o, trainable=False, name="beta_o")


    def save_weights(self, folder_chp):
        self.model_down.qs_net.save_weights(folder_chp+'/checkpoint_qs')
        self.model_down.po_net.save_weights(folder_chp+'/checkpoint_po')
        if self.pi_dim > 0:
            #self.model_top.qpi_net.save_weights(folder_chp+'/checkpoint_qpi')
            self.model_mid.ps_net.save_weights(folder_chp+'/checkpoint_ps')

    def load_weights(self, folder_chp):
        self.model_down.qs_net.load_weights(folder_chp+'/checkpoint_qs')
        self.model_down.po_net.load_weights(folder_chp+'/checkpoint_po')
        if self.pi_dim > 0:
            #self.model_top.qpi_net.load_weights(folder_chp+'/checkpoint_qpi')
            self.model_mid.ps_net.load_weights(folder_chp+'/checkpoint_ps')

    def save_all(self, folder_chp, stats, optimizers={}):
        self.save_weights(folder_chp)
        with open(folder_chp+'/stats.pkl','wb') as ff:
            pickle.dump(stats,ff)
        with open(folder_chp+'/optimizers.pkl','wb') as ff:
            pickle.dump(optimizers,ff)
        copyfile('src/tfmodel.py', folder_chp+'/tfmodel.py')
        copyfile('src/tfloss.py', folder_chp+'/tfloss.py')
        copyfile('src/variables.py', folder_chp+'/variables.py')
        copyfile('src/gym.py', folder_chp+'/gym.py')
        copyfile('src/util.py', folder_chp+'/util.py')

    def load_all(self, folder_chp):
        self.load_weights(folder_chp)
        with open(folder_chp+'/stats.pkl','rb') as ff:
            stats = pickle.load(ff)
        try:
            with open(folder_chp+'/optimizers.pkl','rb') as ff:
                optimizers = pickle.load(ff)
        except:
            optimizers = {}
        if len(stats['var_beta_s'])>0: self.model_down.beta_s.assign(stats['var_beta_s'][-1])
        if len(stats['var_gamma'])>0: self.model_down.gamma.assign(stats['var_gamma'][-1])
        if len(stats['var_beta_o'])>0: self.model_down.beta_o.assign(stats['var_beta_o'][-1])
        return stats, optimizers

    def check_reward(self, o):
        if self.model_down.resolution == 64:
            return tf.reduce_mean(calc_reward(o),axis=[1,2,3]) * 10.0
        elif self.model_down.resolution == 32:
            return tf.reduce_sum(calc_reward_animalai(o), axis=[1,2,3])

    @tf.function
    def imagine_future_from_o(self, o0, pi):
        s0, _, _ = self.model_down.encoder_with_sample(o0)
        ps1, _, _ = self.model_mid.transition_with_sample(pi, s0)
        po1 = self.model_down.decoder(ps1)
        return po1

    @tf.function
    def habitual_net(self, o):
        qs_mean, _ = self.model_down.encoder(o)
        _, Qpi, _ = self.model_top.encode_s(qs_mean)
        return Qpi


           

 
    @tf.function
    def calculate_G_4_repeated_idle(self, o, pi, test,steps=1, calc_mean=False, samples=10):
        """
        We simultaneously calculate G for the four policies of repeating each
        one of the four actions continuously..
        """
        # Calculate current s_t
        
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)
        sum_terms = [tf.zeros([qs0.shape[0]], self.tf_precision), tf.zeros([qs0.shape[0]], self.tf_precision), tf.zeros([qs0.shape[0]], self.tf_precision)]
        sum_G = tf.zeros([qs0.shape[0]], self.tf_precision)
        if calc_mean: 
                    s0_temp_mean = qs0_mean
                    s0_temp_variance=qs0_logvar
                      
        else: s0_temp = qs0
        steps_list=[]
        for t in range(steps):
            if calc_mean:
                G, terms, ps1_mean, po1 = self.calculate_G_mean(s0_temp_mean, pi,test)

            else:
                G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, pi, samples=samples)
                

            sum_terms[0] += terms[0] * (self.discount_factor**t)
            sum_terms[1] += terms[1] * (self.discount_factor**t)
            sum_terms[2] += terms[2] * (self.discount_factor**t)
            sum_G += G * (self.discount_factor**t)
            steps_list.append(sum_G)
            
            if False:
                if t == steps-1:
                    for i in range(5): tf.print("Step: ",t," po1 Action : ", i," ", po1[i])#, output_stream=sys.stdout)

            if calc_mean:
                s0_temp = ps1_mean
            else:
                s0_temp = s1

        return sum_G, sum_terms, po1,steps_list
    

    @tf.function
    def calculate_G_mean(self, s0, pi0,test):
        
        term0 = tf.zeros([s0.shape[0]], self.tf_precision)
        term1 = tf.zeros([s0.shape[0]], self.tf_precision)
        for _ in range(samples):
            
            _, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
            po1 = self.model_down.decoder(ps1_mean)
            _, qs1_mean, qs1_logvar = self.model_down.encoder_with_sample(po1)
            # E [ log P(o|pi) ]
            logpo1 = u.compare_reward(po1)
            #tf.print("logpo1",logpo1)
            
            term0 += logpo1
            
            # E [ log Q(s|pi) - log Q(s|o,pi) ] 

            term1 += -tf.reduce_mean(kl_div_loss_analytically_from_logvar(ps1_mean, ps1_logvar, qs1_mean, qs1_logvar), axis = 1)

        term0 /= float(samples)
        term1 /= float(samples)
        term2_1 = tf.zeros(s0.shape[0], self.tf_precision)
        term2_2 = tf.zeros(s0.shape[0], self.tf_precision)
        temp1_list=[]
        temp2_list=[]
        
        #print("samples in Gmean",samples)
        for _ in range(samples):
            s1=self.model_mid.transition_with_sample(pi0, s0)[0]
            po1_temp1= self.model_down.decoder(s1) #40, 64, 64, 1
            #print("po1_temp1",po1_temp1)
            temp1_list.append(po1_temp1)
            # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
            s1_down=self.model_down.reparameterize(ps1_mean, ps1_logvar)
            po1_temp2= self.model_down.decoder(s1_down) #40, 64, 64, 1
            #print("po1_temp2",po1_temp2)
            temp2_list.append(po1_temp2)
            
        x = tf.stack(temp1_list) 
        y = tf.stack(temp2_list)

        #print("x shape",x.shape) # (10,20,30)
        #print("y shape",y.shape) # (10,20,30)

        term2_1_var = tf.math.reduce_variance(x,0)  #variance of all the 10 samples (20,30)
        term2_1_mean = tf.math.reduce_mean(x,0)  #Mean of all the 10 samples (20,30)
        
        #print("term2_1_var shape",term2_1_var)   
        #print("term2_1_mean shape",term2_1_mean)   
        
        term2_2_var = tf.math.reduce_variance(y,0) 
        term2_2_mean = tf.math.reduce_mean(y,0) 


        term2_1_entropy = entropy_normal_from_logvar(term2_1_var)
        #print("term2_1_entropy",term2_1_entropy) #(20,30)
        term2_1=tf.reduce_sum(term2_1_entropy,1) #axis 1 shape [20]
        term2_2_entropy = entropy_normal_from_logvar(term2_2_var)
        #print("term2_2_entropy",term2_2_entropy)  #(20,30)
        term2_2=tf.reduce_sum(term2_2_entropy,1)  #axis 1  shape [20]
        #print("term2_2",term2_2)
        term2 = term2_1 - term2_2

        #print("term2 shape",term2)   #(10, 64, 9)
        term0 = self.term0_weight * term0
        term1 = self.term1_weight * term1
        term2 = self.term2_weight * term2
        
        if test:
            G = - term0

        else:
            G = - term0 + term1 + term2

        #tf.print("po1 in Gmean",po1)  
        return G, [term0, term1, term2], ps1_mean, po1
    
    @tf.function
    def calculate_G(self, s0, pi0, samples=10):
        term0 = tf.zeros([s0.shape[0]], self.tf_precision)
        #print("term0 first",term0)
        term1 = tf.zeros([s0.shape[0]], self.tf_precision)
        
        for _ in range(samples):

            ps1, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
            po1 = self.model_down.decoder(ps1)
            qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)
            logpo1 =  u.compare_reward(po1, tf.ones_like(po1)) # TODO: np.ones as perfect reward with MSE?
            term0 += logpo1
            term1 += - tf.reduce_mean(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), axis=1)
        term0 /= float(samples)
        term1 /= float(samples)
        
        
        term2_1 = tf.zeros(s0.shape[0], self.tf_precision)
        term2_2 = tf.zeros(s0.shape[0], self.tf_precision)
        
        temp1_list=[]
        temp2_list=[]

        for _ in range(samples):
            # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
            s1=self.model_mid.transition_with_sample(pi0, s0)[1]
            po1_temp1= self.model_down.decoder(s1) #40, 64, 64, 1
            temp1_list.append(po1_temp1)
            # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
            s1_down=self.model_down.reparameterize(ps1_mean, ps1_logvar)
            po1_temp2= self.model_down.decoder(s1_down) #40, 64, 64, 1
            temp2_list.append(po1_temp2)
 
        x = tf.stack(temp1_list)  #shape=(10, 40, 64, 64, 1)  to (10,32,106) 
        term2_1_var=tf.math.reduce_variance(x,0) #shape=(40, 64, 64, 1) to (32, 106)
        term2_1_mean = tf.math.reduce_mean(x,0) 
        term2_1_entropy = entropy_normal_from_logvar(term2_1_var)#shape (40,64,64,1) to (32, 106)
        term2_1=tf.reduce_sum(term2_1_entropy,1) #shape=[20]
 
        
        y = tf.stack(temp2_list)
        
        term2_2_var=tf.math.reduce_variance(y,0) 
        term2_2_entropy = entropy_normal_from_logvar(term2_2_var)
        term2_2=tf.reduce_sum(term2_2_entropy,1)

        
        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2
        G = - term0 + term1 + term2

        
        return G, [term0, term1, term2],s1, ps1_mean,po1
    
