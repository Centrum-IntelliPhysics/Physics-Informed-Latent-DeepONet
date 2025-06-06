import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os

from termcolor import colored
from sklearn import metrics
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12}) 
import seaborn as sns
sns.set_context("paper")
sns.set_style("ticks")

import warnings
warnings.filterwarnings("ignore")


def plot_predictions(i, resultdir, target_i, prediction_i, x_span, inputs_test, X, T, nt, nx, subplot_ylabel, subplot_title, cmap, save, extra_arg=False):

    print(colored('TEST SAMPLE '+str(i+1), 'red'))
    
    r2score = metrics.r2_score(target_i.flatten().cpu().detach().numpy(), prediction_i.flatten().cpu().detach().numpy()) 
    relerror = np.linalg.norm(target_i.flatten().cpu().detach().numpy() - prediction_i.flatten().cpu().detach().numpy()) / np.linalg.norm(target_i.flatten().cpu().detach().numpy())
    r2score = float('%.4f'%r2score)
    relerror = float('%.4f'%relerror)
    print('Rel. L2 Error = '+str(relerror)+', R2 score = '+str(r2score))
    
    fig = plt.figure(figsize=(15,3.5))
    plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)
    
    ax = fig.add_subplot(1, 4, 1)    
    ax.scatter(x_span.cpu().detach().numpy(), inputs_test[i].cpu().detach().numpy(), color='k', s=5)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(subplot_ylabel, rotation="horizontal", fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_title(subplot_title, fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    
    ax = fig.add_subplot(1, 4, 2)  
    cnt = plt.contourf(X.cpu().detach().numpy(), T.cpu().detach().numpy(), target_i.reshape(nt, nx).cpu().detach().numpy(), levels=100, cmap=cmap)
    cnt.set_edgecolor("face")
    cbar = plt.colorbar(cnt, format='%.2f')
    if extra_arg == True:
        cbar.set_ticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$t$', rotation="horizontal", fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_title('True field', fontsize=14)
    ax.tick_params(axis='both', labelsize=10)

    ax = fig.add_subplot(1, 4, 3)  
    cnt = plt.contourf(X.cpu().detach().numpy(), T.cpu().detach().numpy(), prediction_i.reshape(nt, nx).cpu().detach().numpy(), levels=100, cmap=cmap)
    cnt.set_edgecolor("face")
    cbar = plt.colorbar(cnt, format='%.2f')
    if extra_arg == True:
        cbar.set_ticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$t$', rotation="horizontal", fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_title('Predicted field', fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    
    ax = fig.add_subplot(1, 4, 4)  
    cnt = plt.contourf(X.cpu().detach().numpy(), T.cpu().detach().numpy(), np.abs(target_i.reshape(nt, nx).cpu().detach().numpy() - prediction_i.reshape(nt, nx).cpu().detach().numpy()), levels=100, cmap=cmap)
    cnt.set_edgecolor("face")
    plt.colorbar(cnt, format='%.4f')
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$t$', rotation="horizontal", fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_title('Absolute error', fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    
    sns.despine(trim=True)
    plt.tight_layout()

    if save == True:
        plt.savefig(os.path.join(resultdir,'Test_Sample_'+str(i+1)+'.pdf'))
        plt.show()
        plt.close()
    if save == False:
        plt.show()

    print(colored('#'*230, 'green'))


def plot_latentfields(i, resultdir, reconstruction_target_i, latent_target_i, latent_out_i, x_span, inputs_test, z_span, X, T, Z, T_z, nt, nx, latent_dim, subplot_ylabel_1, subplot_title_1, subplot_title_2, cmap, save, extra_arg=False):

    print(colored('TEST SAMPLE '+str(i+1), 'red'))
        
    fig = plt.figure(figsize=(15,3.5))
    plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)
    
    ax = fig.add_subplot(1, 4, 1)    
    ax.scatter(x_span.cpu().detach().numpy(), inputs_test[i].cpu().detach().numpy(), color='k', s=5)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(subplot_ylabel_1, rotation="horizontal", fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_title(subplot_title_1, fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    
    ax = fig.add_subplot(1, 4, 2)  
    cnt = plt.contourf(X.cpu().detach().numpy(), T.cpu().detach().numpy(), reconstruction_target_i.reshape(nt, nx).cpu().detach().numpy(), levels=100, cmap=cmap)
    cnt.set_edgecolor("face")
    cbar = plt.colorbar(cnt, format='%.2f')
    if extra_arg == True:
        cbar.set_ticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$t$', rotation="horizontal", fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_title('True field', fontsize=14)
    ax.tick_params(axis='both', labelsize=10)

    ax = fig.add_subplot(1, 4, 3)  
    plt.pcolor(Z.cpu().detach().numpy(), T_z.cpu().detach().numpy(), latent_target_i.reshape(nt, latent_dim).cpu().detach().numpy(), cmap=cmap)
    plt.colorbar()
    ax.set_xlabel(r'$z$', fontsize=12)
    ax.set_ylabel(r'$t$', rotation="horizontal", fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_xticks(z_span.cpu().detach().numpy())
    ax.set_title(subplot_title_2, fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    
    ax = fig.add_subplot(1, 4, 4)  
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


def plot_AE_reconstructions(i, resultdir_, data_i, reconstructed_i, X, T, cmap, save, extra_arg=False):
        print(colored('TEST SAMPLE '+str(i+1), 'red'))

        mse_i = F.mse_loss(reconstructed_i.cpu(), data_i.cpu())
        print('MSE = {:.2e}'.format(mse_i.item())) # scientific notation

        fig = plt.figure(figsize=(8,3.5))
        plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)

        ax = fig.add_subplot(1, 2, 1)  
        cnt = plt.contourf(X.cpu().detach().numpy(), T.cpu().detach().numpy(), data_i.cpu().detach().numpy(), levels=100, cmap=cmap)
        cnt.set_edgecolor("face")
        cbar = plt.colorbar(cnt, format='%.2f')
        if extra_arg == True:
            cbar.set_ticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$', rotation="horizontal")
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.set_title('True field')

        ax = fig.add_subplot(1, 2, 2)  
        cnt = plt.contourf(X.cpu().detach().numpy(), T.cpu().detach().numpy(), reconstructed_i.cpu().detach().numpy(), levels=100, cmap=cmap)
        cnt.set_edgecolor("face")
        cbar = plt.colorbar(cnt, format='%.2f')
        if extra_arg == True:
            cbar.set_ticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$', rotation="horizontal")
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.set_title('Reconstructed field')  

        sns.despine(trim=True)
        plt.tight_layout()

        if save == True:
            plt.savefig(os.path.join(resultdir_,'Test_Sample_'+str(i+1)+'.pdf'))
            plt.show()
            plt.close()
        if save == False:
            plt.show()

        print(colored('#'*100, 'green'))




