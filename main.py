"""
This software is based on the Chemical Kinetics package by Florent Boudoire (© Copyright 2020, Florent Boudoire, Revision 70db1b3f).
Modifications and enhancements include:
- Implementation of quantitative metrics and QQ-plots
- Simplified data loading process
- Automated figure export system

Original work licensed under the MIT License.
"""
"""
main.py

Main workflow for kinetic fitting.
Contains experiment-specific equations and parameters.
"""
import matplotlib.pyplot as plt
from KineticFitToolkit.dataset import Dataset
from KineticFitToolkit.fitting import fit_dataset
from KineticFitToolkit.plotting import (
    plot_results,
    plot_experimental_and_fit_with_qq,
    plot_qq_plots,
    rmse_mae_plots
)
from KineticFitToolkit.utils import load, select_folder_to_save
from KineticFitToolkit.report import print_result

# ================================================
# EQUATIONS AND SPECIFIC PARAMETERS (TO MODIFY)
# ================================================

def derivatives(y, t, p):
    """System of ODEs for reaction kinetics."""
    c = {"TEOF" : y[0], "I1" : y[1], "I2" : y[2], "I5" : y[3], "P1" : y[4], "P2" : y[5], "EtOH" : y[6]}

    # calculate the differentials
    dc = dict()
    
    dc["TEOF"] =- p["k_4"]*c["TEOF"] + p["k_3"]*c["I1"]*c["EtOH"] 
    
    dc["I1"] = p["k_11"]*c["P1"]*c["EtOH"] - p["k_3"]*c["I1"]*c["EtOH"] + p["k_8"]*c["I2"]*c["EtOH"] + p["k_4"]*c["TEOF"] - p["k_5"]*c["I1"] - p["k_7"]*c["I1"]  
    
    dc["I2"] = + p["k_41"]*c["P1"]  - p["k_9"]*c["I2"] - p["k_21"]*c["I2"] - p["k_116"]*c["I2"] + p["k_7"]*c["I1"] - p["k_8"]*c["I2"]*c["EtOH"] 
    
    dc["I5"] = p["k_130"]*c["P2"] - p["k_120"]*c["I5"] + p["k_116"]*c["I2"] 
          
    dc["P1"] = - p["k_11"]*c["P1"]*c["EtOH"] - p["k_41"]*c["P1"]  + p["k_21"]*c["I2"] - p["k_18"]*c["P1"] + p["k_5"]*c["I1"] + p["k_73"]*c["P2"]*c["EtOH"] 
    
    dc["P2"] = p["k_120"]*c["I5"] + p["k_9"]*c["I2"] + p["k_18"]*c["P1"] - p["k_73"]*c["P2"]*c["EtOH"] - p["k_130"]*c["P2"]
  
    dc["EtOH"] = - p["k_11"]*c["P1"]*c["EtOH"] - p["k_3"]*c["I1"]*c["EtOH"] + p["k_9"]*c["I2"] - p["k_8"]*c["I2"]*c["EtOH"] - p["k_73"]*c["P2"]*c["EtOH"] + p["k_18"]*c["P1"] + p["k_116"]*c["I2"] + p["k_4"]*c["TEOF"] + p["k_5"]*c["I1"] + p["k_7"]*c["I1"] 
   

       
    # dict ("dc") to list ("dy") conversion
    dy = [dc["TEOF"], dc["I1"], dc["I2"], dc["I5"], dc["P1"], dc["P2"], dc["EtOH"]]

    return dy


parameters = {
   "k_3": dict(value = 0.524, min = 0),
   "k_4": dict(value = 0.524, min = 0),
   "k_5": dict(value = 0.158, min = 0),
   "k_11": dict(value = 0.158, min = 0),
   "k_7": dict(value = 0.524, min = 0),
   "k_8": dict(value = 0.524, min = 0),
   "k_9": dict(value = 0.158, min = 0),
   "k_18": dict(value = 0.524, min = 0),
   "k_21": dict(value = 0.524, min = 0),
   "k_41": dict(value = 0.524, min = 0),
   "k_73": dict(value = 0.524, min = 0),
   "k_116": dict(value = 0.524, min = 0),
   "k_120": dict(value = 0.524, min = 0),
   "k_130": dict(value = 0.524, min = 0),
    
}
c0 = {
    "I1": dict(value = 0, vary = False),
    "I2": dict(value = 0, vary = True),
    "I5": dict(value = 0, vary = True),
    "P1": dict(value = 0, vary = True),
    "P2": dict(value = 0, vary = True),
    "EtOH": dict(value = 0, vary = True),
    "TEOF": dict(value = 0.15, vary = True),
    
}
# ================================================
# MAIN WORKFLOW (STABLE)
# ================================================

def main():
    """Executes the full workflow."""
    # Data loading
    file_path = load()
    if not file_path:
        return
        
    save_path = select_folder_to_save()
    dataset = Dataset(file_path)
    
    # Model fitting
    fit_dataset(
        dataset=dataset,
        derivatives=derivatives,
        parameters=parameters,
        c0=c0
    )
    
    # Results generation
    plot_results(dataset, save_path)
    plot_experimental_and_fit_with_qq(dataset, save_path)
    plot_qq_plots(dataset, save_path)
    rmse_mae_plots(dataset, save_path)
    plt.show(block=False)
    input("Press Enter to close the figures...")

    
    # Displaying metrics
    print_result(dataset, save_path)
    print(f'BIC: {dataset.bic_global:.2f}')
    print(f'AIC: {dataset.aic_global:.2f}')

if __name__ == "__main__":
    main()
