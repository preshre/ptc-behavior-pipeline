#!/usr/bin/env python3
"""
Script Helper - AI Assistant for Behavioral Data Analysis Scripts

This script acts as an offline AI agent that explains how to run scripts in the 
Behavior-Data repository and details the required file/folder structure for each script.

Usage:
    python script_helper.py [script_name]
    
Examples:
    python script_helper.py pter_ff.py
    python script_helper.py saa_gs.py
    python script_helper.py --list
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any

# Script metadata containing information about each script
SCRIPT_INFO = {
    "dsaa_ff.py": {
        "title": "DSAA - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for DSAA experiments (CS+ and CS-)",
        "usage": "python dsaa_ff.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "SAA Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """
- Parent Experiment Folder (e.g., PL_CamK2a.4EKD DSAA Freezeframe)
  - Child Experiment CT1 Subfolder (e.g., 20220322 PL_CamK2a.4EKD CT1 DSAA)
    - freeze_SAA1.csv
    - freeze_SAA2.csv
    - ...
  - Child Experiment CT2 Subfolder
  - ...
"""
    },

    "dsaa_gs.py": {
        "title": "DSAA - Graphic State Data Processor",
        "description": "Processes Graphic State data for DSAA experiments (CS+ and CS-)",
        "usage": "python dsaa_gs.py --folder \"/path/to/main_folder\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "Graphic State CSV data files"
        ],
        "folder_structure": """
- Parent Experiment Folder (e.g., PL_SIStag.TSC DSAA)
  - Date ExperimentName CT1 DSAA
    - DSAA1
      - csv files
        - 2023_02_28__16_12_00_A327.csv
        - ...
    - DSAA2
      - csv files
        - ...
  - Date ExperimentName CT2 DSAA
  - ...
"""
    },

    "dtc_ff.py": {
        "title": "DTC - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for DTC experiments (CS+ and CS-)",
        "usage": "python dtc_ff.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "DTC Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """
- Parent Experiment Folder (e.g., PL_CamK2a.4EKD DTC Freezeframe)
  - Child Experiment CT1 Subfolder (e.g., 20220322 PL_CamK2a.4EKD CT1 DTC)
    - freeze_DTC1.csv
    - freeze_DTC2.csv
    - ...
  - Child Experiment CT2 Subfolder
  - ...
"""
    },

    "dtc_unp_ff.py": {
        "title": "Unpaired DTC - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for unpaired DTC protocols",
        "usage": "python dtc_unp_ff.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "DTC Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """
- Parent Experiment Folder (e.g., PL_CamK2a.4EKD DTC_Unpaired Freezeframe)
  - Child Experiment CT1 Subfolder (e.g., 20220322 PL_CamK2a.4EKD CT1 DTC_Unpaired)
    - freeze_DTC1.csv
    - freeze_DTC2.csv
    - ...
  - Child Experiment CT2 Subfolder
  - ...
"""
    },

    "fr_dsaa_plots.py": {
        "title": "Freezing Response - DSAA Plots Generator",
        "description": "Generates plots for DSAA data",
        "usage": "python fr_dsaa_plots.py --folder \"/path/to/processed_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "Processed DSAA data files (.xlsx format)"
        ],
        "folder_structure": """
- Data Folder
  - ProcessedData_DSAA1.xlsx
  - ProcessedData_DSAA2.xlsx
  - ...
"""
    },

    "fr_dtc_plots.py": {
        "title": "Freezing Response - DTC Plots Generator",
        "description": "Generates plots for DTC data",
        "usage": "python fr_dtc_plots.py --folder \"/path/to/processed_data\" --output \"/path/to/output_folder\" --timestamps \"/path/to/timestamps.xlsx\"",
        "required_files": [
            "Processed DTC data files (.xlsx format)",
            "DTC Timestamps.xlsx"
        ],
        "folder_structure": """
- Data Folder
  - ProcessedData_DTC1.xlsx
  - ProcessedData_DTC2.xlsx
  - ...
"""
    },

    "fr_pav_heatmaps.py": {
        "title": "Freezing Response - Pavlovian Heatmaps Generator",
        "description": "Creates heatmaps for Pavlovian conditioning data",
        "usage": "python fr_pav_heatmaps.py --folder \"/path/to/processed_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "Processed Pavlovian data files (.xlsx format)"
        ],
        "folder_structure": """
- Data Folder
  - ProcessedData_PTC1.xlsx
  - ProcessedData_PTC2.xlsx
  - ...
"""
    },

    "fr_ptc_data.py": {
        "title": "Freezing Response - PTC Data Processor",
        "description": "Processes PTC data for visualization",
        "usage": "python fr_ptc_data.py --folder \"/path/to/processed_data\" --output \"/path/to/output_folder\" --timestamps \"/path/to/timestamps.xlsx\"",
        "required_files": [
            "Processed PTC data files (.xlsx format)",
            "PTC Timestamps.xlsx"
        ],
        "folder_structure": """
- Data Folder
  - ProcessedData_PTC1.xlsx
  - ProcessedData_PTC2.xlsx
  - ...
"""
    },

    "fr_ptc_plots.py": {
        "title": "Freezing Response - PTC Plots Generator",
        "description": "Generates plots for PTC data",
        "usage": "python fr_ptc_plots.py --folder \"/path/to/processed_data\" --output \"/path/to/output_folder\" --timestamps \"/path/to/timestamps.xlsx\"",
        "required_files": [
            "Processed PTC data files (.xlsx format)",
            "PTC Timestamps.xlsx"
        ],
        "folder_structure": """
- Data Folder
  - ProcessedData_PTC1.xlsx
  - ProcessedData_PTC2.xlsx
  - ...
"""
    },

    "fr_saa_plots.py": {
        "title": "Freezing Response - SAA Plots Generator",
        "description": "Generates plots for SAA data",
        "usage": "python fr_saa_plots.py --folder \"/path/to/processed_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "Processed SAA data files (.xlsx format)"
        ],
        "folder_structure": """
- Data Folder
  - ProcessedData_SAA1.xlsx
  - ProcessedData_SAA2.xlsx
  - ...
"""
    },

    "generate_summary.py": {
        "title": "Summary Report Generator",
        "description": "Generates summary reports from processed data",
        "usage": "python generate_summary.py --folder \"/path/to/processed_data\" --output \"/path/to/summary_output\"",
        "required_files": [
            "Processed data files (.xlsx format)"
        ],
        "folder_structure": """
- Data Folder
  - ProcessedData1.xlsx
  - ProcessedData2.xlsx
  - ...
"""
    },

    "ptc_ff.py": {
        "title": "PTC - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for PTC (CS+ only)",
        "usage": "python ptc_ff.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "PTC Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """
- Parent Experiment Folder (e.g., PL_CamK2a.4EKD PTC Freezeframe)
  - Child Experiment CT1 Subfolder (e.g., 20220322 PL_CamK2a.4EKD CT1 PTC)
    - freeze_PTC1.csv
    - freeze_PTC2.csv
    - ...
  - Child Experiment CT2 Subfolder
  - ...
"""
    },

    "ptc_staggered_ff.py": {
        "title": "Staggered PTC - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for staggered PTC protocols",
        "usage": "python ptc_staggered_ff.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "PTC Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """
- Parent Experiment Folder (e.g., PL_CamK2a.4EKD PTC_Staggered Freezeframe)
  - Child Experiment CT1 Subfolder (e.g., 20220322 PL_CamK2a.4EKD CT1 PTC_Staggered)
    - freeze_PTC1.csv
    - freeze_PTC2.csv
    - ...
  - Child Experiment CT2 Subfolder
  - ...
"""
    },

    "pter_ff.py": {
        "title": "PTER - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for PTER experiments",
        "usage": "python pter_ff.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "PTER Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """
- Parent Experiment Folder (e.g., PL_CamK2a.4EKD PTER Freezeframe)
  - Child Experiment CT1 Subfolder (e.g., 20220322 PL_CamK2a.4EKD CT1 PTER)
    - freeze_Training.csv  (New: Training phase)
    - freeze_PTE-1.csv
    - freeze_PTE-2.csv
    - freeze_USRein.csv
    - freeze_LTM1d_CtxA.csv
    - freeze_LTM1d_CtxB.csv
    - freeze_LTM1d_CtxD.csv
    - freeze_LTM28d_CtxA.csv
    - freeze_LTM28d_CtxB.csv
    - freeze_LTM28d_CtxD.csv
    - ...
  - Child Experiment CT2 Subfolder
  - ...
"""
    },

    "saa_ff.py": {
        "title": "SAA - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for SAA experiments (CS+ only)",
        "usage": "python saa_ff.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "SAA Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """
- Parent Experiment Folder (e.g., PL_CamK2a.4EKD SAA Freezeframe)
  - Child Experiment CT1 Subfolder (e.g., 20220322 PL_CamK2a.4EKD CT1 SAA)
    - freeze_SAA1.csv
    - freeze_SAA2.csv
    - ...
  - Child Experiment CT2 Subfolder
  - ...
"""
    },

    "saa_ff_postcs.py": {
        "title": "SAA Post-CS - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for SAA with post-CS analysis window",
        "usage": "python saa_ff_postcs.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "SAA Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """Same as saa_ff.py"""
    },

    "saa_ff_postcs15.py": {
        "title": "SAA Post-CS 15s - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for SAA with 15s post-CS analysis window",
        "usage": "python saa_ff_postcs15.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "SAA Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """Same as saa_ff.py"""
    },

    "saa_ff_postcs5.py": {
        "title": "SAA Post-CS 5s - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for SAA with 5s post-CS analysis window",
        "usage": "python saa_ff_postcs5.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "SAA Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """Same as saa_ff.py"""
    },

    "saa_ff_pre_postcs.py": {
        "title": "SAA Pre-Post CS - FreezeFrame Data Processor",
        "description": "Processes FreezeFrame data for SAA with pre and post CS analysis",
        "usage": "python saa_ff_pre_postcs.py --folder \"/path/to/freezeframe_data\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "SAA Timestamps.xlsx (auto-detected)",
            "FreezeFrame CSV data files"
        ],
        "folder_structure": """Same as saa_ff.py"""
    },

    "saa_fibpho_gs.py": {
        "title": "SAA Fiber Photometry - Graphic State Data Processor",
        "description": "Processes fiber photometry Graphic State data for SAA experiments",
        "usage": "python saa_fibpho_gs.py --folder \"/path/to/main_folder\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "Graphic State CSV data files"
        ],
        "folder_structure": """Same as saa_gs.py but with fiber photometry data"""
    },

    "saa_gs.py": {
        "title": "SAA - Graphic State Data Processor",
        "description": "Processes Graphic State data for SAA experiments (CS+ only)",
        "usage": "python saa_gs.py --folder \"/path/to/main_folder\" --output \"/path/to/output_folder\"",
        "required_files": [
            "cohorts.xlsx (auto-detected)",
            "Graphic State CSV data files"
        ],
        "folder_structure": """
- Parent Experiment Folder (e.g., PL_SIStag.TSC SAA)
  - Date ExperimentName CT1 SAA
    - SAA1
      - csv files
        - 2023_02_28__16_12_00_A327.csv
        - ...
    - SAA2
      - csv files
        - ...
  - Date ExperimentName CT2 SAA
  - ...
"""
    },

    "saa_gs_adj.py": {
        "title": "SAA Time-Adjusted - Graphic State Data Processor",
        "description": "Processes time-adjusted Graphic State data for SAA experiments",
        "usage": "python saa_gs_adj.py --folder \"/path/to/main_folder\" --output \"/path/to/output_folder\" --ct \"/path/to/cohorts.xlsx\" --time_discrepancy \"/path/to/time_discrepancy_folder\"",
        "required_files": [
            "cohorts.xlsx",
            "Graphic State CSV data files",
            "Time discrepancy files"
        ],
        "folder_structure": """Same as saa_gs.py with additional time discrepancy folder"""
    },

    "app.py": {
        "title": "Behavior Data Processing Application",
        "description": "Local application interface for data processing",
        "usage": "python app.py",
        "required_files": [
            "All other script files must be available in the same directory"
        ],
        "folder_structure": "N/A"
    },

    "driver.py": {
        "title": "Batch Processing Driver",
        "description": "Driver script for batch processing of multiple datasets",
        "usage": "python driver.py",
        "required_files": [
            "config.json - Configuration file with batch processing details"
        ],
        "folder_structure": "As specified in config.json"
    }
}

class ScriptHelper:
    """AI Assistant for explaining how to run scripts in the Behavior-Data repository."""
    
    def __init__(self):
        """Initialize the ScriptHelper class."""
        self.script_info = SCRIPT_INFO
        self.available_scripts = sorted(list(self.script_info.keys()))
    
    def list_scripts(self) -> None:
        """List all available scripts with a brief description."""
        print("\n=== AVAILABLE SCRIPTS ===\n")
        
        # Group scripts by type
        groups = {
            "FreezeFrame Data Processors": [],
            "Graphic State Data Processors": [],
            "Visualization Tools": [],
            "Utility Scripts": []
        }
        
        for script_name, info in self.script_info.items():
            if "ff" in script_name:
                groups["FreezeFrame Data Processors"].append((script_name, info))
            elif "gs" in script_name:
                groups["Graphic State Data Processors"].append((script_name, info))
            elif "plot" in script_name or "heatmap" in script_name or "fr_" in script_name:
                groups["Visualization Tools"].append((script_name, info))
            else:
                groups["Utility Scripts"].append((script_name, info))
        
        # Print scripts by group
        for group_name, scripts in groups.items():
            if scripts:
                print(f"\n{group_name}:")
                print("-" * len(group_name) + "-" * 2)
                for script_name, info in sorted(scripts):
                    print(f"  {script_name:<25} - {info['description']}")
        
        print("\nFor detailed information about a specific script, use:")
        print("  python script_helper.py <script_name>")
        print("Example: python script_helper.py pter_ff.py")
    
    def get_script_info(self, script_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific script."""
        return self.script_info.get(script_name)
    
    def similar_scripts(self, script_name: str) -> List[str]:
        """Find similar scripts to the requested one."""
        if not script_name:
            return []
        
        # Try to find scripts with similar names
        prefix = script_name.split('_')[0] if '_' in script_name else script_name.split('.')[0]
        return [s for s in self.available_scripts if prefix in s and s != script_name]
    
    def print_script_help(self, script_name: str) -> None:
        """Print detailed help information for a specific script."""
        if script_name not in self.script_info:
            print(f"Error: Script '{script_name}' not found.")
            similar = self.similar_scripts(script_name)
            if similar:
                print("\nDid you mean one of these?")
                for s in similar:
                    print(f"  {s}")
            print("\nUse --list to see all available scripts.")
            return
        
        info = self.script_info[script_name]
        
        print("\n" + "=" * 80)
        print(f"SCRIPT: {script_name}")
        print(f"TITLE: {info['title']}")
        print("=" * 80 + "\n")
        
        print("DESCRIPTION:")
        print(f"  {info['description']}")
        print()
        
        print("USAGE:")
        print(f"  {info['usage']}")
        print()
        
        print("REQUIRED FILES:")
        for item in info['required_files']:
            print(f"  - {item}")
        print()
        
        print("FOLDER STRUCTURE:")
        if info['folder_structure'].strip().startswith("Same as"):
            # Find the referenced script
            ref_script = info['folder_structure'].replace("Same as ", "").strip()
            ref_info = self.script_info.get(ref_script)
            if ref_info:
                print(f"  {ref_info['folder_structure'].strip()}")
            else:
                print(f"  {info['folder_structure'].strip()}")
        else:
            print(f"  {info['folder_structure'].strip()}")
        
        print("\n" + "=" * 80)
        
        # Show related scripts
        similar = self.similar_scripts(script_name)
        if similar:
            print("\nRELATED SCRIPTS:")
            for s in similar:
                print(f"  {s} - {self.script_info[s]['description']}")
            print()

def main():
    """Main function for the script helper."""
    parser = argparse.ArgumentParser(description='AI Assistant for explaining how to run scripts in the Behavior-Data repository')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('script_name', nargs='?', help='Name of the script to get information about')
    group.add_argument('--list', action='store_true', help='List all available scripts')
    args = parser.parse_args()
    
    helper = ScriptHelper()
    
    if args.list:
        helper.list_scripts()
    elif args.script_name:
        helper.print_script_help(args.script_name)
    else:
        print("Welcome to the Behavior-Data Script Helper!")
        print("This AI assistant helps you understand how to run scripts and what file/folder structure is required.")
        print("\nOptions:")
        print("  --list                List all available scripts")
        print("  <script_name>         Get detailed information about a specific script")
        print("\nExample usage:")
        print("  python script_helper.py --list")
        print("  python script_helper.py pter_ff.py")

if __name__ == "__main__":
    main()