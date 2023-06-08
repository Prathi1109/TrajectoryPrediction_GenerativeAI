import matplotlib.pyplot as plt
import numpy as np


left  = 1.0  # the left side of the subplots of the figure
right = 0.5    # the right side of the subplots of the figure
bottom = 0.5   # the bottom of the subplots of the figure
top = 0.6   # the top of the subplots of the figure
wspace = 0.3   # the amount of width reserved for blank space between subplots
hspace = 1.3 # the amount of height reserved for white space between subplots

def stats_plot(stats, filename):
    fig = plt.figure(figsize=(20,18))
    
    plt.subplot(6,5,1)
    plt.plot(np.array(stats['kl_div_s'])+np.array(stats['mse_o']),c='k',label='F')
    plt.plot(np.array(stats['F']),'k--',label='F (weighted)')
    plt.yscale("log")
    plt.title("F")
    #plt.legend()
    #plt.grid(True)

    plt.subplot(6,5,2)
    plt.plot(np.array(stats['F_top']),'k--')#,label='F top')
    plt.yscale("log")
    plt.title("F top")
    #plt.grid(True)

    plt.subplot(6,5,3)
    plt.plot(np.array(stats['F_mid']),'k--')#,label='F mid')
    plt.yscale("log")
    plt.title("F mid")
    #plt.legend()
    #plt.grid(True)

    plt.subplot(6,5,4)
    plt.plot(np.array(stats['F_down']),'k--')#,label='F down')
    plt.yscale("log")
    plt.title("F down")
    #plt.legend()
    #plt.grid(True)

    
    plt.subplot(6,5,5)
    plt.plot(stats['kl_div_s'],'r')#,label='kl_s')
    plt.yscale("log")
    plt.title("KL(s)")
    #plt.grid(True)
    """
    plt.subplot(5,5,6)
    plt.ylabel('KL s dimensions')
    plt.xlabel('epochs')
    for ii in range(len(stats['kl_div_s_anal'][0])):
        if ii < 10:
            plt.plot(np.array(stats['kl_div_s_anal'])[:,ii],label=str(ii))
        else:
            plt.plot(np.array(stats['kl_div_s_anal'])[:,ii])
    plt.legend()

    plt.subplot(5,5,7)
    plt.ylabel('KL s (naive) dimensions')
    plt.xlabel('epochs')
    for ii in range(len(stats['kl_div_s_naive_anal'][0])):
        if ii < 10:
            plt.plot(np.array(stats['kl_div_s_naive_anal'])[:,ii],label=str(ii))
        else:
            plt.plot(np.array(stats['kl_div_s_naive_anal'])[:,ii])
    plt.legend()

    plt.subplot(5,5,8)
    plt.ylabel('Variables')
    for varname in ['a','b','c','beta_s','gamma']:
        plt.plot(np.array(stats['var_'+varname]),label=varname)
    plt.xlabel('epochs')
    plt.yscale("log")
    plt.legend()

    plt.subplot(5,5,9)
    plt.plot(stats['kl_div_pi'],c='y',label='kl_pi')
    plt.yscale("log")
    plt.ylabel('KL(pi)')
    plt.grid(True)

    plt.subplot(5,5,10)
    plt.ylabel('KL pi dimensions')
    plt.xlabel('epochs')
    for ii in range(len(stats['kl_div_pi_anal'][0])):
        if ii < 10:
            plt.plot(np.array(stats['kl_div_pi_anal'])[:,ii],label=str(ii))
        else:
            plt.plot(np.array(stats['kl_div_pi_anal'])[:,ii])
    plt.legend()
    """
    plt.subplot(6,5,6)
    plt.plot(stats['mse_o'],'k')#,label='H(o,P(o))')
    #plt.plot([0,len(stats['mse_o'])],[80.0,80.0],'r--', label='acceptable')
    #plt.plot([0,len(stats['mse_o'])],[60.0,60.0],'g', label='perfect')
    plt.yscale("log")
    plt.title('MSE-Reconstruction')
    #plt.legend()
    #plt.grid(True)

    plt.subplot(6,5,7)
    plt.plot(stats['mse_po1'])
    plt.title('MSE_Prediction')
    plt.xlabel('iterations(x1000)')
    plt.yscale("log")
    #plt.legend()
    """
    plt.subplot(5,5,13)
    plt.plot(stats['mse_po1_mean'])
    plt.ylabel('MSE_po1_mean')
    plt.xlabel('iterations(x1000)')
    plt.yscale("log")
    plt.legend()
    
    plt.subplot(5,5,14)
    plt.plot(stats['mse_po1_sampled'])
    plt.ylabel('MSE_po1_sampled')
    plt.xlabel('iterations(x1000)')
    plt.yscale("log")
    plt.legend()
    
    """
    plt.subplot(6,5,8)
    plt.plot(stats['term0'],c='r')#,label='Extrinsic reward')
    plt.ylabel('term0')
    plt.xlabel('epochs')
    plt.title('Extrinsic reward')
    #plt.yscale("log")
    #plt.legend()

    plt.subplot(6,5,9)
    plt.plot(stats['term1'],c='r')#,label='State Uncertainty')
    plt.ylabel('term1')
    plt.xlabel('epochs')
    plt.title('State Uncertainty')
    #plt.yscale("log")
    #plt.legend()
    
    plt.subplot(6,5,10)
    plt.plot(stats['term2'],c='r')#,label='Model Uncertainty')
    plt.ylabel('term2')
    plt.xlabel('epochs')
    plt.title('Model Uncertainty')
    #plt.yscale("log")
    #plt.legend()
    """
    plt.subplot(5,5,18)
    plt.plot(np.array(stats['omega']),c='b',label='omega')
    plt.plot(np.array(stats['omega'])+np.array(stats['omega_std']),'b--')
    plt.plot(np.array(stats['omega'])-np.array(stats['omega_std']),'b--')
    plt.yscale("log")
    plt.ylabel('omega')
    plt.grid(True)
    """
    plt.subplot(6,5,11)
    plt.plot(stats['total_reward'],'k')#,label='Total reward per game')
    #plt.legend()
    plt.title('Total reward per game')
    #plt.grid(True)
    
    plt.subplot(6,5,12)
    plt.plot(stats['total_timesteps'],'k')#,label='Length per game')
    #plt.legend()
    plt.title('total_timesteps')
    #plt.grid(True)
    
    plt.subplot(6,5,13)
    plt.plot(stats["actual_reward_o1"],'k',label='Actual', color='r', alpha = 0.5)
    plt.plot(stats["expected_reward"],'k',label='Reconstruction', color='b', alpha = 0.5)
    #plt.legend( loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("Actual vs Reconstructed Reward")
    #plt.grid(True)
    
    plt.subplot(6,5,14)
    plt.plot(stats["reward_prediction_error"],'k', color='y', alpha = 0.5)
    plt.ylim(0,1)
    plt.title("Actual vs Reconstructed Reward Error")
    #plt.grid(True)
    
    plt.subplot(6,5,15)
    plt.plot(stats["predicted_reward_mid"],'k',label='Prediction', color='g', alpha = 0.5)
    plt.plot(stats["actual_reward_o1"],'k',label='Actual', color='r', alpha = 0.5)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("Actual vs Predicted Reward")
    #plt.grid(True)
    
    plt.subplot(6,5,16)
    plt.plot(stats["predicted_reward_mid_error"],'k',color='y', alpha = 0.5)
    plt.ylim(0,1)
    plt.title("Actual vs Predicted Reward Error")
    #plt.grid(True)
    
    plt.subplot(6,5,17)
    plt.plot(stats["Actual_X_o1"],'k',label='Actual',  color='r', alpha = 0.5)
    plt.plot(stats["Rec_X"],'k', label='Reconstruction', color='b', alpha = 0.5)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("X position Reconstruction")
    #plt.grid(True)
    
    plt.subplot(6,5,18)
    plt.plot(stats["Actual_X_o1"],'k',label='Actual',  color='r', alpha = 0.5)
    plt.plot(stats["Pre_X"],'k',label='Prediction', color='g', alpha = 0.5)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("X position Prediction")
    #plt.grid(True)
    
    plt.subplot(6,5,19)
    plt.plot(stats["Actual_Y_o1"],'k', label='Actual',  color='r', alpha = 0.5)
    plt.plot(stats["Rec_Y"],'k', label='Reconstruction',color='b', alpha = 0.5)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("Y position Reconstruction")
    #plt.grid(True)
    
    plt.subplot(6,5,20)
    plt.plot(stats["Actual_Y_o1"],'k', label='Actual',  color='r', alpha = 0.5)
    plt.plot(stats["Pre_Y"],'k',label='Prediction',color='g', alpha = 0.5)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("Y position Prediction")
    #plt.grid(True)
    
    plt.subplot(6,5,21)
    plt.plot(stats["Actual_vel_X_o1"],'k', label='Actual', color='r', alpha = 0.5)
    plt.plot(stats["Rec_vel_X"],'k',label='Reconstruction', color='b', alpha = 0.5)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("Velocity_X Reconstruction")
    #plt.grid(True)
    
    plt.subplot(6,5,22)
    plt.plot(stats["Actual_vel_X_o1"],'k', label='Actual', color='r', alpha = 0.5)
    plt.plot(stats["Pre_vel_X"],'k',label='Prediction', color='g', alpha = 0.5)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("Velocity_X Prediction")
    #plt.grid(True)
    
    plt.subplot(6,5,23)
    plt.plot(stats["Actual_vel_Y_o1"],'k', label='Actual', color='r', alpha = 0.5)
    plt.plot(stats["Rec_vel_Y"],'k',label='Reconstruction', color='b', alpha = 0.5)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("Velocity_Y Reconstruction")
    #plt.grid(True)
    
    plt.subplot(6,5,24)
    plt.plot(stats["Actual_vel_Y_o1"],'k', label='Actual', color='r', alpha = 0.5)
    plt.plot(stats["Pre_vel_Y"],'k',label='Prediction', color='g', alpha = 0.5)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title("Velocity_Y Prediction")
    #plt.grid(True)
    
    plt.subplot(6,5,25)
    plt.plot(stats['kl_div_pi'],c='y',label='kl_pi')
    plt.yscale("log")
    plt.ylabel('KL(pi)')
    plt.title("KL Divergence pi")
    #plt.grid(True)
    """
    plt.subplot(6,5,26)
    plt.ylabel('KL pi dimensions')
    plt.xlabel('epochs')
    
    for ii in range(len(stats['kl_div_pi_anal'][0])):
        if ii < 10:
            plt.plot(np.array(stats['kl_div_pi_anal'])[:,ii],label=str(ii))
        else:
            plt.plot(np.array(stats['kl_div_pi_anal'])[:,ii])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Selected Actions")
    """
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    fig.set_tight_layout(True)
    plt.savefig(filename+'.png')
    
    
    
    
    plt.close()
