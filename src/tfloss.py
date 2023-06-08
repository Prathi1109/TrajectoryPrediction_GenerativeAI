import tensorflow as tf
import numpy as np
import src.util as u
from src.tfutils import *
import tensorflow_probability as tfp
from src.variables import *

def compute_omega(kl_pi, a, b, c, d):
    return a * ( 1.0 - 1.0/(1.0 + np.exp(- (kl_pi-b) / c)) ) + d



@tf.function
def compute_loss_mid(model_mid, s0, Ppi_sampled, qs1_mean, qs1_logvar, omega):
    ps1, ps1_mean, ps1_logvar = model_mid.transition_with_sample(Ppi_sampled, s0)

    # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
    # ----------------------------------------------------------------------
    print("omega",omega)
    kl_div_s_anal = kl_div_loss_analytically_from_logvar(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar)
    kl_div_s = tf.reduce_mean(kl_div_s_anal, 1)

    F_mid = kl_div_s
    loss_terms = (kl_div_s, kl_div_s_anal)
    return F_mid, loss_terms, ps1, ps1_mean, ps1_logvar

@tf.function
def compute_loss_down(model_down, o1, ps1_mean, ps1_logvar, omega, displacement = 0.00001, scale = 100, return_mean = False):
    
    # Reconstruction loss
    qs1_mean, qs1_logvar = model_down.encoder(o1)
    
    if True: 
        qs1 = model_down.reparameterize(qs1_mean, qs1_logvar)
    else: # deterministic model down
        qs1 = qs1_mean
        
    po1 = model_down.decoder(qs1)
    logpo1_s1 = -tf.reduce_mean(tf.math.squared_difference(po1*scale, o1*scale), axis=[1]) 
    
    po1_mean = model_down.decoder(qs1_mean)
    
    if False: 
        po1_pred = model_down.decoder(ps1_mean)
        logpo1_s1_pred = -tf.reduce_mean(tf.math.squared_difference(o1*scale, po1_pred*scale), axis = [1]) 
        tf.print("REC:\nO\t", o1[0][-1], " \nP\t", po1[0][-1], " \nE\t",logpo1_s1[0])
        tf.print("PRED:\nVAR\t", qs1_logvar[0], "\nV_P\t", ps1_logvar[0], "\nMEAN\t", qs1_mean[0], " \nME_P\t", ps1_mean[0], " \nO\t", o1[0], " \nP\t", po1_pred[0], " \nE\t",logpo1_s1_pred[0])
        tf.print("PRED:\nVAR\t", qs1_logvar[3], "\nV_P\t", ps1_logvar[3], "\nMEAN\t", qs1_mean[3], " \nME_P\t", ps1_mean[3], " \nO\t", o1[3], " \nP\t", po1_pred[3], " \nE\t",logpo1_s1_pred[3])
    omega=tf.cast(omega, tf.float32)
    tf.print("omega",omega)
    kl_div_s_naive_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, 0.5, 0.0, omega)
    kl_div_s_naive = tf.reduce_mean(kl_div_s_naive_anal, 1)

    kl_div_s_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega)
    kl_div_s = tf.reduce_mean(kl_div_s_anal, 1)

    #F = - logpo1_s1 + model_down.gamma * 1 * kl_div_s + (1-model_down.gamma) * 100 * kl_div_s_naive #10
    #F = -logpo1_s1 + model_down.gamma * 1000 * kl_div_s_naive # with annealed KL_naive
    F = -logpo1_s1 + 10 * kl_div_s #10 * kl_div_s_naive + 
    
    if False:
        tf.print("kl_div_s", kl_div_s)
        tf.print("kl_div_s_naive", kl_div_s_naive)
        tf.print("logpo1_s1", logpo1_s1)
        tf.print("model_down.gamma", model_down.gamma)
    
    loss_terms = (-logpo1_s1, kl_div_s, kl_div_s_anal, kl_div_s_naive, kl_div_s_naive_anal)
    if not return_mean:
        return F, loss_terms, po1,  qs1
    else:
        return F, loss_terms, po1,  qs1, po1_mean
     

@tf.function
def train_model_mid(model_mid, s0, qs1_mean, qs1_logvar, Ppi_sampled, omega, optimizer):
    s0_stopped = tf.stop_gradient(s0)
    qs1_mean_stopped = tf.stop_gradient(qs1_mean)
    qs1_logvar_stopped = tf.stop_gradient(qs1_logvar)
    Ppi_sampled_stopped = tf.stop_gradient(Ppi_sampled)
    omega_stopped = tf.stop_gradient(omega)
    with tf.GradientTape() as tape:
        F, loss_terms, ps1, ps1_mean, ps1_logvar = compute_loss_mid(model_mid=model_mid, s0=s0_stopped, Ppi_sampled=Ppi_sampled_stopped, qs1_mean=qs1_mean_stopped, qs1_logvar=qs1_logvar_stopped, omega=omega_stopped)
        gradients = tape.gradient(F, model_mid.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_mid.trainable_variables))
    return ps1_mean, ps1_logvar

@tf.function
def train_model_down(model_down, o1, ps1_mean, ps1_logvar, omega, optimizer):
    ps1_mean_stopped = tf.stop_gradient(ps1_mean)
    ps1_logvar_stopped = tf.stop_gradient(ps1_logvar)
    omega_stopped = tf.stop_gradient(omega)
    with tf.GradientTape() as tape:
        F, _, _, _ = compute_loss_down(model_down=model_down, o1=o1, ps1_mean=ps1_mean_stopped, ps1_logvar=ps1_logvar_stopped, omega=omega_stopped)
        gradients = tape.gradient(F, model_down.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_down.trainable_variables))
