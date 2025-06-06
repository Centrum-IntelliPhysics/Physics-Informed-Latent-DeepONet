import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns
sns.set_context("paper")
sns.set_style("ticks")

def plot_source(sample_index, x_span, y_span, source_value, title, cmap, resultdir, save=False):
    x_span, y_span, source_value = x_span.cpu().detach().numpy(), y_span.cpu().detach().numpy(), source_value.cpu().detach().numpy()
    
    levels = np.arange(0, 1.1, 0.1)

    # Plot the result as a contour plot
    plt.figure(figsize=(6, 4))
    cp = plt.contourf(x_span, y_span, source_value, levels = levels, cmap=cmap, vmin=0, vmax=1)  # Restrict color bar range
    plt.colorbar(cp)  # Add a color bar

    # Labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title, color='red')
    
    sns.despine(trim=True)
    plt.tight_layout()

    if save == True:
        plt.savefig(os.path.join(resultdir,'Test_Sample_'+str(sample_index+1)+'_Source.pdf'))
        plt.show()
        plt.close()
    if save == False:
        plt.show()

def plot_solution(sample_index, x, y, solution, time_stamps, title, cmap, fontsize, levels, resultdir, save=False, filename=None):
    x, y, solution, time_stamps = x.cpu().detach().numpy(), y.cpu().detach().numpy(), solution.cpu().detach().numpy(), time_stamps.cpu().detach().numpy()
    
    desired_time_stamps = np.arange(0.1, 1.1, 0.1)
    # Find indices of the closest time stamps
    indices = [np.abs(time_stamps - t).argmin() for t in desired_time_stamps]
    desired_solution = solution[indices, :]
    
    fig, axs = plt.subplots(3, 4, figsize=(18, 10))
    fig.suptitle(title, fontsize=fontsize*2, y=1.0, color='red')
    axs = axs.flatten()

    # Loop through time points and plot the solution for each time
    for i, t in enumerate(desired_time_stamps):
        cf = axs[i].contourf(x, y, desired_solution[i], 
                             levels=levels, cmap=cmap, 
                             vmin=desired_solution.min(), vmax=desired_solution.max())
        for b in cf.collections:
            b.set_edgecolor("face")
        cbar = plt.colorbar(cf, ax=axs[i])
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f')) 
        axs[i].set_title(f't = {t:.2f}', fontsize=fontsize+2)
        axs[i].set_xlabel(r'$x$', fontsize=fontsize)
        axs[i].set_ylabel(r'$y$', rotation = 0, fontsize=fontsize, labelpad=15)  
        axs[i].tick_params(axis='both', which='major', labelsize=fontsize-2)

    # Hide the last 3 subplots (since there are only 5 time points)
    for i in range(len(desired_time_stamps), len(axs)):
        axs[i].axis('off')

    sns.despine(trim=True)
    plt.tight_layout()
    
    if save == True:
        plt.savefig(os.path.join(resultdir,'Test_Sample_'+str(sample_index+1)+'_'+filename+'.pdf'))
        plt.show()
        plt.close()
    if save == False:
        plt.show()