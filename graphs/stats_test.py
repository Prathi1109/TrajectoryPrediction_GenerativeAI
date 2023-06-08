import matplotlib.pyplot as plt
import numpy as np


left  = 1.0  # the left side of the subplots of the figure
right = 0.5    # the right side of the subplots of the figure
bottom = 0.5   # the bottom of the subplots of the figure
top = 0.6   # the top of the subplots of the figure
wspace = 0.3   # the amount of width reserved for blank space between subplots
hspace = 1.3 # the amount of height reserved for white space between subplots


def stats_plot_test(stats, filename):
    fig = plt.figure(figsize=(16,12))
    
    plt.subplot(4,4,1)
    plt.plot(stats['test_total_reward'],'k')#,label='Total reward per game')
    #plt.legend()
    plt.title('Total reward per game')
    plt.grid(True)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    fig.set_tight_layout(True)
    plt.savefig(filename+'.png')
    plt.close()

