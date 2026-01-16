import pandas as pd
from typing import Dict, Optional
import warnings
import traceback
import freezeframe as ff
import logging
import re
import os
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

class FRPTC(ff.FreezeFrame):
    def _get_timestamps_path(self) -> str:
        """walk through parent directory of self.folder to find the timestamps file"""
        parent = self.folder_path.parent
        for root, dirs, files in os.walk(parent):
            for file in files:
                if file.endswith('.xlsx') and 'timestamps' in file.lower():
                    return os.path.join(root, file)
        raise FileNotFoundError("Timestamps file not found")
    
    def process_sheets(self) -> Dict:
        """Process all timestamp sheets"""
        logging.info("Processing timestamp sheets")
        try:
            # Read the Excel file without headers
            self.timestamps_path = self._get_timestamps_path()
            df = pd.read_excel(self.timestamps_path, header=None)
            
            # Initialize the result dictionary
            result = {}
            
            # Current section tracker
            current_section = None
            
            # Iterate through the rows
            for idx, row in df.iterrows():
                # First value in the row (column 0)
                first_val = str(row[0]).strip() if pd.notna(row[0]) else ""
                
                # Check if this is a section header (Training or LTM)
                if first_val in ['Training', 'LTM']:
                    current_section = first_val
                    result[current_section] = {}
                
                # Skip header rows and empty rows
                elif (current_section is not None and 
                        first_val not in ['', 'Epoch', 'nan'] and 
                        pd.notna(row[0])):
                    
                    # Extract CS number and epoch type
                    match = re.match(r'(Pre-|Post-)?CS(\d+)', first_val)
                    if match:
                        prefix = match.group(1) or ''  # Pre-, Post-, or ''
                        cs_num = match.group(2)  # The CS number
                        
                        # Initialize CS number dict if it doesn't exist
                        if cs_num not in result[current_section]:
                            result[current_section][cs_num] = {}
                        
                        # Determine epoch type
                        epoch_type = prefix + 'CS' if prefix else 'CS'
                        
                        # Add the epoch data to the current section
                        # Column 1 is Onset, Column 2 is Offset
                        result[current_section][cs_num][epoch_type] = {
                            "Onset": int(row[1]) if pd.notna(row[1]) else None,
                            "Offset": int(row[2]) if pd.notna(row[2]) else None
                        }
            
            logging.info(f"Processed all timestamps successfully")
            return result
            
        except Exception as e:
            logging.error(f"Error processing timestamp sheets: {str(e)}", exc_info=True)
            raise

    def get_cols(self, num_of_cs):
        top_level = ['Animal ID']
        for i in range(1, num_of_cs + 1):
            top_level.extend([f'CS{i}'] * 3)  # Repeat each CS number 3 times
            
        # Create second level with Pre-CS, CS, Post-CS repeated for each CS number
        second_level = [''] + ['Pre-CS', 'CS', 'Post-CS'] * num_of_cs
        
        return pd.MultiIndex.from_arrays([top_level, second_level])
    
    def process_experiment(self, ff_df, experiment_name, ct):
        experiment_name = experiment_name.upper()
        logging.info(f"Processing experiment: {experiment_name} in {ct}")

        # Pre-calculate experiment timestamps
        exp_timestamps = self.timestamps["Training" if "training" in experiment_name.lower() else "LTM"]
        
        num_of_cs = len(exp_timestamps.keys())
        # Initialize DataFrame with pre-allocated size
        df = pd.DataFrame(columns=self.get_cols(num_of_cs))
        data_rows = []
        
        for animal_id in ff_df.iloc[1:, 0]:
            try:
                sorted_cs = sorted(exp_timestamps.keys(), key=int)

                data = []

                for cs in sorted_cs:
                    periods = ['Pre-CS', 'CS', 'Post-CS']

                    for period in periods:
                        if period in exp_timestamps[cs]:
                            start, end = exp_timestamps[cs][period].values()
                            data.append(self.get_ff_avg(animal_id, start, end, ff_df))
                
                data_rows.append([animal_id] + data)

            except KeyError as e:
                logging.warning(f'Error processing animal {animal_id} in {ct}: {str(e)}')
                continue

        # Bulk append all rows at once
        if data_rows:
            df = pd.concat([df, pd.DataFrame(data_rows, columns=self.get_cols(num_of_cs))], 
                          ignore_index=True)

        logging.info(f"Completed processing experiment: {experiment_name} in {ct}")
        return df
        
    def parse_sheet(self, xlsx: pd.ExcelFile, sheet: str) -> Optional[Dict]:
        return super().parse_sheet(xlsx, sheet)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PTC FreezeFrame data and extract metrics')
    parser.add_argument('--folder', type=str, help='Path to the root directory containing the various experiments\' data')
    parser.add_argument('--output', type=str, help='Path to the output directory for plots and logs')
    parser.add_argument('--timestamps', type=str, help='Path to the Excel file with timestamps (if not auto-detected)')
    args = parser.parse_args()
    
    # Configure logging
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output, 'ptc_data_processing.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting PTC data processing")
    logging.info(f"Input folder: {args.folder}")
    logging.info(f"Output folder: {args.output}")
    
    try:
        fr_ptc = FRPTC(folder_path=Path(args.folder), output_path=Path(args.output), timestamps_path=args.timestamps)
        fr_ptc.main()
    except Exception as e:
        logging.critical(f"Critical failure in execution: {e}")
        traceback.print_exc()