"""
utils.py

Utility functions for file dialogs and file/folder selection.
"""

import tkinter as tk
from tkinter import filedialog
import os

def load():
    """Open a file dialog to select a CSV file."""
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=(("CSV files", "*.csv"),)
    )
    if file:
        print(f"Selected file: {file}")
        return file
    else:
        print("No file selected")
        return None

def select_folder_to_save():
    """Open a dialog to select a folder for saving results."""
    save_path = filedialog.askdirectory(title="Select a folder to save")
    if save_path:
        if os.path.exists(save_path):
            print(f"Selected folder: {save_path}")
            return save_path
        else:
            print("The folder does not exist.")
            return None
    else:
        print("No folder selected.")
        return save_path
