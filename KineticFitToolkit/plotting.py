"""
plotting.py

Contains plotting functions for visualizing experimental data,
fitted curves, Q-Q plots, and error metrics.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import statsmodels.api as sm



def plot_results(dataset, save_path):
    """Plot experimental vs fitted data and R² scores with y-axis fixed from 0 to 1."""
    plt.figure(figsize=[20, 12], dpi=150)
    handles_labels = []
    for i, column in enumerate(dataset.df_c.columns[1:]):
        line, = plt.plot(dataset.df_c['t'], dataset.df_c[column], 'o', markersize=8,
                        label=f'{column} $R^2$ = {dataset.r2_scores[column]:.2f}', 
                        color=f'C{i}')
        handles_labels.append((line, column, dataset.r2_scores[column]))
    for i, column in enumerate(dataset.df_c_fit.columns[1:]):
        plt.plot(dataset.df_c_fit['t'], dataset.df_c_fit[column], linewidth=2, color=f'C{i}')
    sorted_handles_labels = sorted(handles_labels, key=lambda x: x[1])
    handles, labels = zip(*[
    (h, f'{l} $R^2$ = {r:.2f}') for h, l, r in sorted_handles_labels])
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1, fontsize=14, frameon=False)
    plt.xlabel('Time (min)', fontsize=14)
    plt.ylabel('Concentration (M)', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.title("Concentration Profiles of Species Over Time", fontsize=16, pad=20)
    plt.subplots_adjust(left=0.15, right=0.70)
    plt.savefig(os.path.join(save_path, "general_profile.tiff"), format="tiff", dpi=100, bbox_inches='tight')
    
    # Plot R² scores with y-axis from 0 to 1
    plt.figure(figsize=[15, 12], dpi=150)
    names = list(dataset.r2_scores.keys())
    r2_values = list(dataset.r2_scores.values())
    sorted_pairs = sorted(zip(names, r2_values), key=lambda x: x[0])
    names, r2_values = zip(*sorted_pairs)
    bars = plt.bar(names, r2_values, color='teal')
    plt.xticks(range(len(names)), names, fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.ylim(0, 1.05)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, val in zip(bars, r2_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.2f}', ha='center', va='bottom', fontsize=14)
    plt.title("$R^2$ Values for Product Concentration Profiles", fontsize=16, pad=20)
    
    plt.savefig(os.path.join(save_path, "R_squared.tiff"), format="tiff", dpi=100, bbox_inches='tight')
    
    

def plot_experimental_and_fit_with_qq(dataset, save_path):
    """Plots experimental data, fitted values, and Q-Q plots with confidence intervals for each column."""
    def sort_key(col):
        try:
            return int(col)
        except ValueError:
            return float('inf')
    sorted_columns = sorted(dataset.df_c.columns[1:], key=sort_key)
    r2_scores = dataset.r2_scores
    num_columns = len(sorted_columns)
    fig, axs = plt.subplots(num_columns, 2, figsize=(40, 40 * num_columns), 
                       gridspec_kw={'hspace': 0.7, 'wspace': 0.3})
    for i, column in enumerate(sorted_columns):
        axs[i, 0].plot(dataset.df_c['t'], dataset.df_c[column], 'o', markersize=10, 
                      label=f'{column} $R^2$ = {r2_scores[column]:.2f}', 
                      color=f'C{i}')
        axs[i, 0].plot(dataset.df_c_fit['t'], dataset.df_c_fit[column], 
                      linewidth=3, color=f'C{i}')
        #axs[i, 0].set_xlabel('Time (min)', fontsize=12)
        #axs[i, 0].set_ylabel('Concentration (M)', fontsize=12)
        #axs[i, 0].set_title(f'Fit Results for {column}', fontsize=12)
        axs[i, 0].legend(fontsize=12)
        axs[i, 0].grid(True)
        axs[i, 0].tick_params(axis='both', labelsize=12)
        f_interp = interp1d(dataset.df_c_fit['t'], dataset.df_c_fit[column], fill_value="extrapolate")
        y_pred = f_interp(dataset.df_c['t'])
        y_true_sorted = np.sort(dataset.df_c[column])
        y_pred_sorted = np.sort(y_pred)
        model = sm.OLS(y_true_sorted, sm.add_constant(y_pred_sorted)).fit()
        y_fit = model.predict(sm.add_constant(y_pred_sorted))
        residuals = y_true_sorted - y_fit
        std_err = np.std(residuals)
        ci_upper = y_fit + 1.96 * std_err
        ci_lower = y_fit - 1.96 * std_err
        axs[i, 1].scatter(y_pred_sorted, y_true_sorted, color='blue', s=70)
        axs[i, 1].plot([min(y_pred_sorted), max(y_pred_sorted)], [min(y_pred_sorted), max(y_pred_sorted)], color='red', linestyle='--', linewidth=3)
        axs[i, 1].fill_between(y_pred_sorted, ci_lower, ci_upper, color='lightblue', alpha=0.5)
        axs[i, 1].grid(True)
        axs[i, 1].tick_params(axis='both', labelsize=12)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.suptitle("Kinetic Profiles and Model Fit Quality for the Product Distribution", fontsize=16)
    plt.savefig(os.path.join(save_path, "fit_qq_plots.tiff"), dpi=100, bbox_inches='tight', format='tiff')
    
def plot_qq_plots(dataset, save_path):
    """Plots Q-Q plots for each species in ascending numerical order with three columns."""
    def sort_key(col):
        try:
            return int(col)
        except ValueError:
            return float('inf')
    sorted_columns = sorted(dataset.df_c.columns[1:], key=sort_key)
    num_columns = len(sorted_columns)
    num_rows = (num_columns + 2) // 3
    fig, axs = plt.subplots(num_rows, 3, figsize=(18, 7 * num_rows))
    axs = axs.flatten()
    for i, col in enumerate(sorted_columns):
        f_interp = interp1d(dataset.df_c_fit['t'], dataset.df_c_fit[col], fill_value="extrapolate")
        y_pred = f_interp(dataset.df_c['t'])
        y_true_sorted = np.sort(dataset.df_c[col])
        y_pred_sorted = np.sort(y_pred)
        model = sm.OLS(y_true_sorted, sm.add_constant(y_pred_sorted)).fit()
        y_fit = model.predict(sm.add_constant(y_pred_sorted))
        residuals = y_true_sorted - y_fit
        axs[i].scatter(y_pred_sorted, y_true_sorted,color='blue', s=70)
        axs[i].plot(y_pred_sorted, y_pred_sorted, 'r--', lw=3)
        axs[i].fill_between(y_pred_sorted, y_fit - 1.96*np.std(residuals), y_fit + 1.96*np.std(residuals), color='lightblue', alpha=0.5)
        axs[i].set_title(f'Q-Q Plot: {col}', fontsize=14)
        axs[i].set_xlabel('Theoretical Quantiles', fontsize=12)
        axs[i].set_ylabel('Experimental Quantiles', fontsize=12)
        axs[i].tick_params(labelsize=10)
        axs[i].grid(True)
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    plt.subplots_adjust(hspace=0.4, wspace=0.2)  # Contrôle précis de l'espacement 
    plt.savefig(os.path.join(save_path, "QQ_plots.tiff"), dpi=100, bbox_inches='tight', format='tiff')
    
   
def rmse_mae_plots(dataset, save_path):
    """Plot RMSE and MAE error metrics as bar charts."""
    sorted_keys = sorted(dataset.relative_rmse.keys(), key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
    rmse_values = [dataset.relative_rmse[k] for k in sorted_keys]
    mae_values = [dataset.relative_mae[k] for k in sorted_keys]
    x = np.arange(len(sorted_keys)) * 0.7
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 8), dpi=150)
    rects1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', color='#c44e52', alpha=0.8, edgecolor='white')
    rects2 = ax.bar(x + width/2, mae_values, width, label='MAE', color='#dd8452', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Compounds', fontsize=14)
    ax.set_ylabel('Error Metrics (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_keys, fontsize=14)
    ax.legend(fontsize=14, frameon=False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=14)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)
    autolabel(rects1)
    autolabel(rects2)
    plt.title("RMSE and MAE Values for Product Concentration Profiles", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "error_metrics.tiff"), dpi=100, bbox_inches='tight', format='tiff')
    
    return
