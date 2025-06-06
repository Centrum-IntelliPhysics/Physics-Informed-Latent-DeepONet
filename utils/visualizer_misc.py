import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
# import pandas as pd

from matplotlib.ticker import FuncFormatter
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12}) 
import seaborn as sns
sns.set_theme(context='paper', style='ticks', rc=plt.rcParams)

import warnings
warnings.filterwarnings("ignore")

def plot_loss_terms(resultdir, iteration_list, loss_list, datadriven_loss_list, pinn_loss_list, save):
    plt.figure()
    plt.plot(iteration_list, loss_list, 'g', label='total loss')
    plt.plot(iteration_list, datadriven_loss_list, 'r', label='data-driven loss')
    plt.plot(iteration_list, pinn_loss_list, 'b', label='pinn loss')
    plt.yscale("log")
    plt.gca().xaxis.get_major_formatter()._useMathText = True 
    plt.xlabel('Iterations')
    plt.ylabel('Training loss')
    plt.legend(loc="best", frameon=False)
    sns.despine(trim=True)
    plt.tight_layout(pad=0.1)
    if save == True:
        plt.savefig(os.path.join(resultdir, 'loss_terms_plot.pdf'))  


def plot_training_loss(resultdir, iteration_list, loss_list, save):
    plt.figure()
    plt.plot(iteration_list, loss_list, 'g', label = 'training loss')
    plt.yscale("log")
    plt.gca().xaxis.get_major_formatter()._useMathText = True 
    plt.xlabel('Iterations')
    plt.ylabel('Training loss')
    sns.despine(trim=True)
    plt.tight_layout(pad=0.1)
    if save == True:
        plt.savefig(os.path.join(resultdir,'loss_plot.pdf'))


def plot_testing_loss(resultdir, test_iteration_list, test_loss_list, save):
    plt.figure()
    plt.scatter(test_iteration_list, test_loss_list,  color='r', label = 'testing loss')
    plt.yscale("log")
    plt.gca().xaxis.get_major_formatter()._useMathText = True 
    plt.xlabel('Iterations')
    plt.ylabel('Testing loss')
    sns.despine(trim=True)
    plt.tight_layout(pad=0.1)
    if save == True:
        plt.savefig(os.path.join(resultdir,'test_loss_plot.pdf'))


def plot_training_testing_loss(resultdir, iteration_list, loss_list, test_iteration_list, test_loss_list, save):
    plt.figure()
    plt.plot(iteration_list, loss_list, 'g', label='training loss')
    plt.scatter(test_iteration_list, test_loss_list,  color='r', label = 'testing loss')
    plt.yscale("log")
    plt.gca().xaxis.get_major_formatter()._useMathText = True 
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc="best", frameon=False)
    sns.despine(trim=True)
    plt.tight_layout(pad=0.1)
    if save == True:
        plt.savefig(os.path.join(resultdir, 'train_test_plot.pdf'))  


def plot_learningrates(resultdir, iteration_list, learningrates_list, save):
    plt.figure()
    plt.plot(iteration_list, learningrates_list, 'b', label = 'learning-rate')
    plt.gca().xaxis.get_major_formatter()._useMathText = True 
    plt.xlabel('Iterations')
    plt.ylabel('Learning-rate')
    sns.despine(trim=True)
    plt.tight_layout(pad=0.1)
    if save == True:
        plt.savefig(os.path.join(resultdir,'learning-rate_plot.pdf'))


def plot_box(df, x_col, y_col, hue_col, xlabel, ylabel, title, save=False, save_path=None, filename=None, custom_format=False, annotate=False, notation=None, scale=1e-4, group_labels=None, category_order=None, figsize=(8, 6)):
    """
    Create a box plot for y_col data.

    Parameters:
    - df: DataFrame containing the data to plot.
    - x_col: Column name for the x-axis.
    - y_col: Column name for the y-axis.
    - hue_col: Column name for hue (categorical variable).
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - save: Boolean indicating whether to save the plot.
    - save_path: Path to save the plot if `save` is True.
    - filename: Filename to save the plot if `save` is True.
    - custom_format: Boolean indicating whether to apply a custom formatter to the y-axis.
    - annotate: Boolean indicating whether to add the notation to the y-axis.
    - notation: The notation to display on the y-axis (e.g., scientific notation '1e-4').
    - scale: The scaling factor for custom formatting (default is 1e-4).
    - group_labels: List of tuples with group information [(start_idx, end_idx, label), ...].
    - category_order: Define the order of categories for the x-axis.
    - figsize: Tuple specifying the figure size (default is (8, 6)).
    """
    
    def custom_formatter(x, pos):
        return f'{x/scale:.1f}'

    # Define the colors manually
    color_palette = {"PI-Vanilla-NO": "#1f77b4",  # Blue color
                     "PI-Latent-NO": "#d62728"}  # Red color

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=x_col, y=y_col, data=df, hue=hue_col, ax=ax, order=category_order, palette=color_palette, width=0.5)
    ax.set_xlabel(xlabel, fontsize=14*1.25)
    ax.set_ylabel(ylabel, fontsize=14*1.25)
    ax.set_title(title, fontsize=16*1.25)

    ax.tick_params(axis='both', labelsize=14*1.25)

    if custom_format:
        # Apply the custom formatter to the y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    
    if annotate and notation:
        # Add the notation to the top of the y-axis
        ax.annotate(rf'${notation}$', xy=(0, 1), xycoords='axes fraction', fontsize=14*1.5,
                    xytext=(-10, 15), textcoords='offset points',
                    ha='center', va='center')

    add_vlines = True
    # fixed_y_position = df[y_col].max()  # 10% above the maximum value across all plots
    # Add group subheadings
    if group_labels:
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        fixed_y_position = ymax + 0.025 * y_range  # 5% padding above plot
        for start_idx, end_idx, label in group_labels:
            x_position = (start_idx + end_idx) / 2.0  # Center position for the label
            ax.text(x_position, fixed_y_position, label, ha='center', va='bottom', fontsize=12, color='black')

            # Add vertical dotted line if enabled
            if add_vlines and end_idx < len(df[x_col].unique()) - 1:
                ax.axvline(x=end_idx + 0.5, color='black', linewidth=1, linestyle='-')
    
    legend = ax.legend(fontsize=12)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, filename))
        plt.show()
        plt.close()
    else:
        plt.show()


def plot_violin(df, x_col, y_col, hue_col, xlabel, ylabel, title, save=False, save_path=None, filename=None, custom_format=False, annotate=False, notation=None, scale=1e-4, limit_yaxis=False, yaxis_starting_limit=None):
    """
    Create a violin plot for y_col data.

    Parameters:
    - df: DataFrame containing the data to plot.
    - x_col: Column name for the x-axis.
    - y_col: Column name for the y-axis.
    - hue_col: Column name for hue (categorical variable).
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - save: Boolean indicating whether to save the plot.
    - save_path: Path to save the plot if `save` is True.
    - filename: Filename to save the plot if `save` is True.
    - custom_format: Boolean indicating whether to apply a custom formatter to the y-axis.
    - annotate: Boolean indicating whether to add the notation to the y-axis.
    - notation: The notation to display on the y-axis (e.g., scientific notation '1e-4').
    - scale: The scaling factor for custom formatting (default is 1e-4).
    - limit_yaxis: Boolean indicating whether to limit y-axis.
    - yaxis_starting_limit: Set y-axis to start from this limit.
    """
    
    def custom_formatter(x, pos):
        return f'{x/scale:.1f}'

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x=x_col, y=y_col, data=df, hue=hue_col, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if custom_format:
        # Apply the custom formatter to the y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    
    if annotate and notation:
        # Add the notation to the top of the y-axis
        ax.annotate(rf'${notation}$', xy=(0, 1), xycoords='axes fraction', fontsize=10,
                    xytext=(-10, 15), textcoords='offset points',
                    ha='center', va='center')

    # Set y-axis to start from yaxis_starting_limit
    if limit_yaxis:
        ax.set_ylim(yaxis_starting_limit)
        
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, filename))
        plt.show()
        plt.close()
    else:
        plt.show()

# The function to handle scientific notation for MSE and integer formatting for training time
def calculate_mean_std_scientific_ver0(df, group_cols, target_cols, scale_factors, rename_dict):
    results = {}

    for target_col in target_cols:
        scaled_col = df[target_col] * scale_factors.get(target_col, 1)
        grouped = df.groupby(group_cols)[scaled_col.name].agg(['mean', 'std']).reset_index()

        # Apply different formatting based on the column name
        if target_col == 'Mean Squared Error Test':
            # Format MSE in scientific notation
            grouped[target_col] = grouped.apply(
                lambda row: f"{row['mean']:.1e} ± {row['std']:.1e}", axis=1
            )
        elif target_col == 'Training Time (in sec)':
            # Format training time as integers
            grouped[target_col] = grouped.apply(
                lambda row: f"{int(row['mean'])} ± {int(row['std'])}", axis=1
            )

        results[target_col] = grouped.drop(columns=['mean', 'std'])

    # Merge results from all target columns
    merged_df = results[target_cols[0]]
    for target_col in target_cols[1:]:
        merged_df = merged_df.merge(results[target_col], on=group_cols)

    merged_df = merged_df.rename(columns=rename_dict)
    return merged_df

# The function to handle scientific notation for MSE and integer formatting for training time
def calculate_mean_std_scientific(df, group_cols, target_cols, scale_factors, format_spec, rename_dict):
    results = {}

    for target_col in target_cols:
        scaled_col = df[target_col] * scale_factors.get(target_col, 1)
        grouped = df.groupby(group_cols)[scaled_col.name].agg(['mean', 'std']).reset_index()

        # Apply formatting
        if target_col in format_spec:
            fmt = format_spec[target_col]
            grouped[target_col] = grouped.apply(
                lambda row: f"{format(row['mean'], fmt)} ± {format(row['std'], fmt)}", axis=1
            )

        results[target_col] = grouped.drop(columns=['mean', 'std'])

    # Merge results from all target columns
    merged_df = results[target_cols[0]]
    for target_col in target_cols[1:]:
        merged_df = merged_df.merge(results[target_col], on=group_cols)

    merged_df = merged_df.rename(columns=rename_dict)
    return merged_df

# Function to generate LaTeX table from DataFrame
def generate_latex_table(df):
    # Start the LaTeX table
    latex_table = r"\begin{tabular}{rlll}" + "\n" + r"\hline" + "\n"
    
    # Add the header
    latex_table += r"   $\mathrm{Model}$ & $\mathrm{n}_{\mathrm{train}}$ & $\mathrm{R^2\ Score\ Test}$ & $\mathrm{Rel.\ L2\ Error\ Test}$ & $\mathrm{Training\ Time\ (in\ sec)}$ & $\mathrm{Runtime\ per\ Iteration\ (in\ sec/iter)}$ \\" + "\n" + r"\hline" + "\n"
    
    # Add the rows of the DataFrame
    for index, row in df.iterrows():
        # Format each row as LaTeX table row
        latex_table += f" {row['Model']} &{row['n_train']} & {row['R2 score Test']} & {row['Rel. L2 Error Test']} & {row['Training Time (in sec)']} & {row['Runtime per Iteration (in sec/iter)']} \\\\ \n"
    
    # Close the table
    latex_table += r"\hline" + "\n" + r"\end{tabular}"

    return latex_table

