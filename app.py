#!/usr/bin/env python3
"""
Entry point for the Behavior Data Analysis application.
This script initializes and launches the UI.

Developed by Harshil Sanghvi for Shrestha Lab
"""
import os
import sys
import tkinter as tk
from experiment_manager.gui import ExperimentManagerGUI
from experiment_manager.utils import check_requirements

# Add signature as a constant
SIGNATURE = "Developed by Harshil Sanghvi for Shrestha Lab"

def main():
    """Main function to launch the application"""
    # Print signature banner
    print("\n" + "="*80)
    print(f"  {SIGNATURE}  ".center(80, "*"))
    print("="*80 + "\n")
    
    if not check_requirements():
        sys.exit(1)
    
    # Launch the GUI
    root = tk.Tk()
    app = ExperimentManagerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()