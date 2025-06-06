import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
from termcolor import colored
import seaborn as sns
sns.set_context("paper")
sns.set_style("ticks")

def plot_input_field(sample_index, x_span, y_span, input_field, title, cmap, fontsize, levels, resultdir, save=False):

    # Plot the result as a contour plot
    plt.figure(figsize=(6, 4))
    cp = plt.contourf(x_span, y_span, input_field, levels=levels, cmap=cmap)  
    plt.colorbar(cp)  # Add a color bar

    # Labels and title
    plt.xlabel('x')
    plt.ylabel('y', rotation = 0)
    plt.title(title, color='red', fontsize=fontsize)
    
    sns.despine(trim=True)
    plt.tight_layout()

    if save == True:
        plt.savefig(os.path.join(resultdir,'Test_Sample_'+str(sample_index+1)+'_Inputfield.pdf'))
        plt.show()
        plt.close()
    if save == False:
        plt.show()

def plot_solution_field(sample_index, x, y, solution, time_stamps, title, cmap, fontsize, levels, resultdir, save=False, filename=None):
    
    desired_time_stamps = np.arange(0., 1.1, 0.1)
    # Find indices of the closest time stamps
    indices = [np.abs(time_stamps - t).argmin() for t in desired_time_stamps]
    desired_solution = solution[indices, :]
    
    fig, axs = plt.subplots(3, 4, figsize=(18, 10))
    fig.suptitle(title, fontsize=fontsize*2, y=1.0, color='red')
    axs = axs.flatten()

    # Loop through time points and plot the solution for each time
    for i, t in enumerate(desired_time_stamps):
        cf = axs[i].contourf(x, y, desired_solution[i], 
                             levels=levels, cmap=cmap)
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

def plot_latentfields(i, resultdir, latent_target_i, latent_out_i, z_span, Z, T_z, nt, latent_dim, subplot_title_2, cmap, save, extra_arg=False):

    print(colored('TEST SAMPLE '+str(i+1), 'red'))
        
    fig = plt.figure(figsize=(15,3.5))
    plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)

    ax = fig.add_subplot(1, 2, 1)  
    plt.pcolor(Z.cpu().detach().numpy(), T_z.cpu().detach().numpy(), latent_target_i.reshape(nt, latent_dim).cpu().detach().numpy(), cmap=cmap)
    plt.colorbar()
    ax.set_xlabel(r'$z$', fontsize=12)
    ax.set_ylabel(r'$t$', rotation="horizontal", fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_xticks(z_span.cpu().detach().numpy())
    ax.set_title(subplot_title_2, fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    
    ax = fig.add_subplot(1, 2, 2)  
    plt.pcolor(Z.cpu().detach().numpy(), T_z.cpu().detach().numpy(), latent_out_i.reshape(nt, latent_dim).cpu().detach().numpy(), cmap=cmap)
    plt.colorbar()
    ax.set_xlabel(r'$z$', fontsize=12)
    ax.set_ylabel(r'$t$', rotation="horizontal", fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_xticks(z_span.cpu().detach().numpy())
    ax.set_title('Predicted latent field', fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    
    sns.despine(trim=True)
    plt.tight_layout()

    if save == True:
        plt.savefig(os.path.join(resultdir,'Test_Sample_'+str(i+1)+'_latent.pdf'))
        plt.show()
        plt.close()
    if save == False:
        plt.show()

    print(colored('#'*230, 'green'))