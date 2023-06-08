import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def reconstructions_plot(o0, o1, po1,term2_1_var,term2_2_var, filename, colour=False,train=False):
    if colour:
        o0 = o0[:3,:]
        o1 = o1[:3,:]
        po1 = po1[:3,:]
    else:
        o0 = o0[:3,:]
        o1 = o1[:3,:]
        po1 = po1[:3,:]
    fig = plt.figure(figsize=(5,20))
    print("o0 shape",o0.shape)
    plt.subplot(10,1,1)
    ax0 = sns.heatmap(o0[0].reshape(16,1))
    ax0.tick_params(left=False, bottom=False)
    ax0.set(xticklabels=[])
    ax0.set(yticklabels=[])
    ax0.set(xlabel=None)
    plt.ylabel('o0_0')
    
    plt.subplot(10,1,2)
    ax0 = sns.heatmap(o0[1].reshape(16,1))
    ax0.tick_params(left=False, bottom=False)
    ax0.set(xticklabels=[])
    ax0.set(yticklabels=[])
    ax0.set(xlabel=None)
    plt.ylabel('o0_1')
    
    plt.subplot(10,1,3)
    ax0 = sns.heatmap(o1[0].reshape(16,1))
    ax0.tick_params(left=False, bottom=False)
    ax0.set(xticklabels=[])
    ax0.set(yticklabels=[])
    ax0.set(xlabel=None)
    plt.ylabel('o1_0')
    
    plt.subplot(10,1,4)
    ax0 = sns.heatmap(o1[0].reshape(16,1))
    ax0.tick_params(left=False, bottom=False)
    ax0.set(xticklabels=[])
    ax0.set(yticklabels=[])
    ax0.set(xlabel=None)
    plt.ylabel('o1_1')
    
    plt.subplot(10,1,5)
    ax0 = sns.heatmap(po1[0].reshape(16,1))
    ax0.tick_params(left=False, bottom=False)
    ax0.set(xticklabels=[])
    ax0.set(yticklabels=[])
    ax0.set(xlabel=None)
    plt.ylabel('po1_0')
    
    plt.subplot(10,1,6)
    ax0 = sns.heatmap(po1[0].reshape(16,1))
    ax0.tick_params(left=False, bottom=False)
    ax0.set(xticklabels=[])
    ax0.set(yticklabels=[])
    ax0.set(xlabel=None)
    plt.ylabel('po1_1')
    
    
    if train:
    
        plt.subplot(10,1,7)
        ax0 = sns.heatmap(term2_1_var[0].reshape(16,1))
        ax0.tick_params(left=False, bottom=False)
        ax0.set(xticklabels=[])
        ax0.set(yticklabels=[])
        ax0.set(xlabel=None)
        plt.ylabel('ax0_21')

        plt.subplot(10,1,8)
        ax1 = sns.heatmap(term2_1_var[1].reshape(16,1))
        ax1.tick_params(left=False, bottom=False)
        ax1.set(xticklabels=[])
        ax1.set(yticklabels=[])
        ax1.set(xlabel=None)
        plt.ylabel('ax1_21')
        
        plt.subplot(10,1,9)
        ax3 = sns.heatmap(term2_2_var[0].reshape(16,1))
        ax3.tick_params(left=False, bottom=False)
        ax3.set(xticklabels=[])
        ax3.set(yticklabels=[])
        ax3.set(xlabel=None)
        plt.ylabel('ax0_22')

        plt.subplot(10,1,10)
        ax4 = sns.heatmap(term2_2_var[1].reshape(16,1))
        ax4.tick_params(left=False, bottom=False)
        ax4.set(xticklabels=[])
        ax4.set(yticklabels=[])
        ax4.set(xlabel=None)
        plt.ylabel('ax1_22')


    fig.set_tight_layout(True)
    
    plt.savefig(filename)
    plt.close()