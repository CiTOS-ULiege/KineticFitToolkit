"""
fitting.py

Contains functions for fitting kinetic models to experimental data,
including the main fit_dataset function and helpers for evaluating the model.
"""

import lmfit
import pandas as pd
import numpy as np
from .utils import load, select_folder_to_save

def fit_dataset(dataset, derivatives, parameters, c0={}):
    df_c = dataset.df_c
    tracked_species = dataset.names
    params = lmfit.Parameters()
    for key, value in parameters.items():
        params.add(key, **value)
    for name in tracked_species:
        key = fr"c0_{name}"
        default_val = df_c[name][0] if name in df_c else 0
        if name in c0:
            params.add(key, **c0[name])
        else:
            params.add(key, value=default_val, vary=True)
    dataset.init_params = params
    result = lmfit.minimize(
        residuals,
        params,
        args=(df_c, derivatives, tracked_species),
        nan_policy='omit'
    )
    print(result.message)
    dataset.fit_result = result
    t_fit = np.linspace(df_c["t"].min(), df_c["t"].max(), 150)
    c_fit = evaluate(derivatives, result.params, t_fit)
    dataset.df_c_fit = pd.DataFrame({"t": t_fit})
    for i, s in enumerate(tracked_species):
        dataset.df_c_fit[s] = c_fit[:, i]
    dataset.r2_scores = dataset.calculate_r2_scores()
    dataset.rmse_scores = dataset.calculate_mean_squared_errors()
    dataset.mae_scores = dataset.calculate_mean_absoolute_errors()
    dataset.rmse_mae_ratio = dataset.calculate_rmse_mae_ratio()
    dataset.concentration_xp_max = dataset.calculate_concentration_xp_max()
    dataset.relative_rmse = dataset.calculate_relative_rmse()
    dataset.relative_mae = dataset.calculate_relative_mae()
    dataset.residuals_2 = dataset.calculate_residuals_2()
    dataset.aic_global, dataset.bic_global = dataset.calculate_global_AIC_BIC(parameters)

def evaluate(derivatives, params, t):
    from scipy.integrate import odeint
    c0 = [params[key].value for key in params if "c0_" in key]
    p = {key: params[key].value for key in params if "c0_" not in key}
    c = odeint(
        func=derivatives,
        y0=c0,
        t=t,
        args=(p,)
    )
    return c

def calculate_residuals(df, fit, names):
    res = []
    if len(fit.shape) == 1:
        fit = fit.reshape(-1, 1)
    for i, name in enumerate(names):
        norm = df[name] + fit[:, i]
        partial_res = (df[name] - fit[:, i]) / norm
        idx_div0 = norm == 0
        partial_res[idx_div0] = 0
        res.extend(partial_res)
    return res

def residuals(params, df_c, derivatives, tracked_species): 
    c = evaluate(
        derivatives=derivatives,
        params=params,
        t=df_c["t"]
    )
    res = calculate_residuals(df_c, c, tracked_species)
    return res
