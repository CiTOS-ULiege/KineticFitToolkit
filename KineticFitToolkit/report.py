"""
report.py

Prints and exports the fitting results in a formatted DataFrame.
"""

import pandas as pd
import os

def print_result(dataset, save_path):
    """Prints the fitting results in a formatted DataFrame."""
    params = dataset.fit_result.params
    init_params = dataset.init_params
    r2_scores = dataset.r2_scores
    rmse_scores = dataset.rmse_scores
    mae_scores = dataset.mae_scores
    rmse_mae_ratio = dataset.rmse_mae_ratio
    clean = lambda value: f"{value:.3g}" if value is not None else None
    data = []
    for p in params:
        compound_name = p.replace("c0_", "")
        r2_value = r2_scores.get(compound_name, None)
        mse_value = rmse_scores.get(compound_name, None)
        mae_value = mae_scores.get(compound_name, None)
        rmse_mae_ratio_value = rmse_mae_ratio.get(compound_name, None)
        data.append([
            p,
            clean(params[p].value),
            clean(init_params[p].value),
            clean(r2_value), 
            clean(mse_value),
            clean(mae_value),
            clean(rmse_mae_ratio_value),
        ])
    df = pd.DataFrame(
        data=data,
        columns=["Name", "Fitted Value", "Initial Value", "R²", "RMSE", "MAE", "RMSE/MAE"]
    )
    aic_bic_row = pd.DataFrame({
        'Name': ['Global AIC', 'Global BIC'],
        'Fitted Value': [clean(dataset.aic_global), clean(dataset.bic_global)]
    })
    df_with_aic_bic = pd.concat([df, aic_bic_row], ignore_index=True)
    df_with_aic_bic.to_excel(os.path.join(save_path, "results.xlsx"), index=False)
    print(df)
    return df
