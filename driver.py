import json
import subprocess
import os
import glob
import argparse
from pathlib import Path

# Add signature as a constant
SIGNATURE = "Developed by Harshil Sanghvi for Shrestha Lab"


def sanitize_json(json_path):
    """Fix common issues with JSON files created by copy-pasting paths."""
    try:
        with open(json_path, 'r') as f:
            content = f.read()
        
        # Replace single quotes with double quotes
        if "'" in content:
            content = content.replace("'", '"')
            
            # Save the sanitized content back
            with open(json_path, 'w') as f:
                f.write(content)
            print(f"Fixed quote issues in {json_path}")
    except Exception as e:
        print(f"Error sanitizing JSON: {e}")


def load_config(config_path):
    """Load and parse the configuration file."""
    # First sanitize the JSON file
    sanitize_json(config_path)
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Please check your config.json file for formatting errors.")
        return []


def expand_folder_paths(params):
    """Expand folder paths to handle directories with multiple experiment subfolders."""
    expanded_params = []
    folder_path = params['folder']
    
    # Check if the folder is actually a directory of experiments
    if params.get('is_experiment_collection', False):
        # Get all subdirectories
        subdirs = [d for d in glob.glob(os.path.join(folder_path, "*")) if os.path.isdir(d)]
        
        if not subdirs:
            print(f"Warning: No subdirectories found in {folder_path}")
            return []
        
        # Create a parameter set for each subdirectory
        for subdir in subdirs:
            new_params = params.copy()
            new_params['folder'] = subdir
            new_params.pop('is_experiment_collection', None)  # Remove the flag
            expanded_params.append(new_params)
    else:
        # Just a single experiment folder
        expanded_params.append(params)
        
    return expanded_params


def run_scripts(param_combinations, output_base_folder, dry_run=False):
    """Execute each script with its parameters."""
    all_expanded_params = []
    
    # First, expand any experiment collections
    for params in param_combinations:
        all_expanded_params.extend(expand_folder_paths(params))
    
    if not all_expanded_params:
        print("No valid parameter combinations found.")
        return
    
    print(f"Found {len(all_expanded_params)} experiment(s) to process.")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)
    
    # Now run each expanded parameter set
    for idx, params in enumerate(all_expanded_params, 1):
        folder_name = os.path.basename(params['folder'])
        script_name = os.path.basename(params['script'])
        output_folder = os.path.join(output_base_folder, folder_name)

        if script_name == 'generate_summary.py':
            output_folder = params['folder']
        
        # Build the command
        cmd = [
            'python', params['script'],
            '--folder', params['folder'],
            '--output', output_folder
        ]
        
        # Add optional parameters if present
        for param_name in ['timestamps', 'ct']:
            if param_name in params:
                cmd.extend([f'--{param_name}', params[param_name]])
        
        # Add any other custom parameters
        for key, value in params.items():
            if key not in ['script', 'folder', 'timestamps', 'ct', 'is_experiment_collection']:
                cmd.extend([f'--{key}', str(value)])
        
        print(f"\n[{idx}/{len(all_expanded_params)}] Running: {script_name} on {folder_name}")
        print(f"Command: {' '.join(cmd)}")
        
        if dry_run:
            print("Dry run mode - command not executed")
        else:
            try:
                subprocess.run(cmd, check=True)
                print(f"✓ Successfully processed: {folder_name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Error processing {folder_name}: {e}")


def main():
    # Print signature banner
    print("\n" + "="*80)
    print(f"  {SIGNATURE}  ".center(80, "*"))
    print("="*80 + "\n")
    
    parser = argparse.ArgumentParser(description="Run analysis scripts on experiment folders.")
    parser.add_argument("--config", default="config.json", help="Path to the configuration file")
    parser.add_argument("--output", default="output", help="Base output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.config}")
    param_combinations = load_config(args.config)
    
    if param_combinations:
        run_scripts(param_combinations, args.output, args.dry_run)
    else:
        print("No valid configuration found. Exiting.")
        
    # Print signature footer
    print("\n" + "="*80)
    print(f"  {SIGNATURE}  ".center(80, "*"))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()