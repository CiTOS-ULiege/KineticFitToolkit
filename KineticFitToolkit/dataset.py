"""
dataset.py

Contains the Dataset class for loading and handling experimental data,
as well as methods for calculating fit metrics (R², RMSE, MAE, etc.).
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Dataset:
    def __init__(self, files_c, t_label="t [s]", c_label="C [M]"):
        """Initialize Dataset with experimental CSV file."""
        self.df_c = None
        self.df_c_fit = None
        self.t_label = t_label
        self.c_label = c_label
        self.names = None
        self.fit_results = None
        self.init_params = None
        self.r2_scores = None  
        self.rmse_scores = None
        self.mae_scores = None
        self.rmse_mae_ratio = None
        self.concentration_xp_max = None
        self.relative_rmse = None
        self.relative_mae = None
        self.residuals_2 = None
        self.aic_global = None
        self.bic_global = None
        self.load_c(files_c)
        
    def load_c(self, file):
        """Load data from a file and store in df_c."""
        self.df_c = pd.read_csv(file)
        self.names = [name for name in self.df_c.columns if name != "t"]
    
    def calculate_r2_scores(self, kind='quadratic'):
        """Calculate R² scores for fitted parameters."""
        scores = {}
        if self.df_c_fit is None:
            raise ValueError("Fitted data (df_c_fit) is not available. Please run the fit first.")
        for col in self.names:
            f_interp = interpolate.interp1d(self.df_c_fit['t'], self.df_c_fit[col], kind=kind, fill_value="extrapolate")
            col_interp = f_interp(self.df_c['t'])
            score = r2_score(self.df_c[col], col_interp)
            scores[col] = score
        self.r2_scores = scores
        return scores

    def calculate_mean_squared_errors(self):
        """Calculate RMSE for fitted parameters."""
        rmse_scores = {}
        if self.df_c_fit is None:
            raise ValueError("Fitted data (df_c_fit) is not available. Please run the fit first.")
        for col in self.names:
            f_interp = interpolate.interp1d(self.df_c_fit['t'], self.df_c_fit[col], kind='linear', fill_value="extrapolate")
            col_interp = f_interp(self.df_c['t'])
            rmse_score = np.sqrt(mean_squared_error(self.df_c[col], col_interp))
            rmse_scores[col] = rmse_score
        self.rmse_scores = rmse_scores
        return rmse_scores

    def calculate_mean_absoolute_errors(self):
        """Calculate MAE for fitted parameters."""
        mae_scores = {}
        if self.df_c_fit is None:
            raise ValueError("Fitted data (df_c_fit) is not available. Please run the fit first.")
        for col in self.names:
            f_interp = interpolate.interp1d(self.df_c_fit['t'], self.df_c_fit[col], kind='linear', fill_value="extrapolate")
            col_interp = f_interp(self.df_c['t'])
            mae_score = mean_absolute_error(self.df_c[col], col_interp)
            mae_scores[col] = mae_score
        self.mae_scores = mae_scores
        return mae_scores

    def calculate_rmse_mae_ratio(self):
        """Calculate the ratio between RMSE and MAE for each compound."""
        rmse_mae_ratio = {}
        for compound in self.names:
            if self.mae_scores[compound] != 0:
                rmse_mae_ratio[compound] = self.rmse_scores[compound] / self.mae_scores[compound]
            else:
                rmse_mae_ratio[compound] = float('inf')
        return rmse_mae_ratio    

    def calculate_concentration_xp_max(self):
        """Calculate the maximum experimental concentration for each compound."""
        concentration_xp_max = {}
        for col in self.names:
            max_concentration = self.df_c[col].max()         
            concentration_xp_max[col] = max_concentration
        self.concentration_xp_max = concentration_xp_max
        return concentration_xp_max

    def calculate_relative_rmse(self):
        """Calculate relative RMSE as a percentage of max concentration."""
        relative_rmse = {}
        for compound in self.names:
            relative_rmse[compound] = (self.rmse_scores[compound] / self.concentration_xp_max[compound])*100
        self.relative_rmse = relative_rmse
        return relative_rmse

    def calculate_relative_mae(self):
        """Calculate relative MAE as a percentage of max concentration."""
        relative_mae = {}
        for compound in self.names:
            relative_mae[compound] = (self.mae_scores[compound] / self.concentration_xp_max[compound])*100
        self.relative_mae = relative_mae
        return  relative_mae

    def calculate_residuals_2(self):
        """Calculate residuals between experimental and fitted data."""
        residuals_2 = {}
        if self.df_c_fit is None:
            raise ValueError("Fitted data (df_c_fit) is not available. Please run the fit first.")
        for col in self.names:
            f_interp = interpolate.interp1d(self.df_c_fit['t'], self.df_c_fit[col], kind='linear', fill_value="extrapolate")
            col_interp = f_interp(self.df_c['t'])
            residual_2 = self.df_c[col] - col_interp
            residual_2 = np.array(residual_2)
            residuals_2[col] = residual_2
        self.residuals_2 = residuals_2
        return residuals_2

    def calculate_global_AIC_BIC(self, parameters):
        """Calculate global AIC and BIC for the fit."""
        if self.residuals_2 is None:
            self.calculate_residuals_2()
        n = len(self.df_c) * len(self.names)
        k = sum(1 for param in parameters if param.startswith("k_"))
        print("Number of constants is", k)
        sse_total = sum(np.sum(residual**2) for residual in self.residuals_2.values())
        aic_global = 2*k + n * np.log(sse_total/n)
        bic_global = np.log(n)*k + n * np.log(sse_total/n)
        self.aic_global = aic_global
        self.bic_global = bic_global
        return aic_global, bic_global
