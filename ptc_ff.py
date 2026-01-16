import pandas as pd
import numpy as np
import os
from pandas import ExcelWriter
import warnings
import sys
import re
import argparse
import logging
from datetime import datetime
from utils import setup_logging

warnings.filterwarnings("ignore")

class FreezeFrame:
    def __init__(self, folder_path, output_path):
        '''Function to initialize the class with the paths to the timestamps file, CT file, folder containing the FreezeFrame data, and the output folder.'''
        self.folder_path = folder_path
        self.output_path = output_path
        self.ct_path = self.get_ct_path()
        self.timestamps_path = self.get_timestamps_path()
        self.training_timestamps, self.ltm_timestamps = None, None
        self.output = self.output_path
        self.experiment_name = None
        self.current_ct = None
        setup_logging(self.output_path)
        logging.info(f"FreezeFrame initialized with:\n\tTimestamps: {self.timestamps_path}\n\tCT: {self.ct_path}\n\tFolder: {folder_path}\n\tOutput: {output_path}")
        
    @staticmethod
    def parse_arguments(description="Process FreezeFrame data"):
        '''Function to parse the command line arguments.'''
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing the FreezeFrame data')
        parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
        return parser.parse_args()
    
    def get_cols(self, experiment):
        '''Function to get the column names for the given experiment.'''
        logging.debug(f"Getting columns for experiment {experiment}")
        if experiment == 7:
            return pd.MultiIndex.from_arrays([['Animal ID', ' ', ' '] + ['CS+']*4 + ['ITI']*3 + [' '],
                                          ['', 'Threshold', 'Pre-CS'] + [str(i) for i in range(1, 4)] + ['Mean CS+'] + [str(i) for i in range(1, 3)] + ['Mean ITI', 'Post-CS']])
        
        return pd.MultiIndex.from_arrays([['Animal ID', ' ', ' '] + ['CS+']*3 + ['ITI']*2 + [' '],
                                          ['', 'Threshold', 'Pre-CS'] + [str(i) for i in range(1, 3)] + ['Mean CS+'] + [str(i) for i in range(1, 2)] + ['Mean ITI', 'Post-CS']])
        

    def process_timestamps(self):
        '''Function to process the timestamps file and extract the training and LTM timestamps.'''
        logging.info(f"Processing timestamps file: {self.timestamps_path}")
        df = pd.read_excel(self.timestamps_path)
        logging.debug(f"Timestamps file loaded with shape: {df.shape}")

        # Find the index where "Epoch" occurs to split the DataFrame
        split_index = df.index[df["Unnamed: 0"] == "Epoch"].tolist()
        logging.debug(f"Split indices found at: {split_index}")

        # Split the DataFrame into two based on the split index
        training_df = df.iloc[2:split_index[1]-1, :].reset_index(drop=True)
        ltm_df = df.iloc[split_index[1]:, :].reset_index(drop=True)

        # Drop the unnecessary rows with NaN values in the first column
        training_df = training_df.dropna(subset=['Unnamed: 0'], how='all')
        ltm_df = ltm_df.dropna(subset=['Unnamed: 0'], how='all')

        # Set column names
        training_df.columns = training_df.iloc[0]
        ltm_df.columns = ltm_df.iloc[0]

        # Drop the first row as it's just a repetition of column names
        training_df = training_df.iloc[1:].reset_index(drop=True)
        ltm_df = ltm_df.iloc[1:].reset_index(drop=True)

        self.training_timestamps = training_df
        self.ltm_timestamps = ltm_df
        logging.info(f"Timestamps processed - Training shape: {training_df.shape}, LTM shape: {ltm_df.shape}")
    
    def get_cohort_data(self, ct):
        '''Function to extract the cohort data from the CT file.'''
        logging.info(f"Getting cohort data for CT: {ct}")
        df = pd.read_excel(self.ct_path, usecols=range(5))
        
        ct_row_index = df.index[df.iloc[:, 0].str.contains(ct, na=False)].tolist()[0]
        logging.debug(f"Found CT row index: {ct_row_index}")

        # Extract rows following the CT row until a row with all NaN values is encountered
        new_df_rows = []
        for i in range(ct_row_index+1, len(df)):
            if df.iloc[i].isnull().all():
                break
            new_df_rows.append(df.iloc[i])

        # Create a new DataFrame with the extracted rows
        new_df = pd.DataFrame(new_df_rows[1:])
        new_df.columns = new_df_rows[0].tolist()

        # drop first col
        new_df.drop(new_df.columns[0], axis=1, inplace=True)

        new_df.reset_index(drop=True, inplace=True) # reset index
        new_df.columns = pd.MultiIndex.from_arrays([new_df.columns, [''] + ['']*len(new_df.columns[1:])]) # set multi-level columns
        new_df_renamed = new_df.rename(columns={'Animal': 'Animal ID'}) # rename columns
        logging.debug(f"Cohort data extracted with shape: {new_df_renamed.shape}")
        return new_df_renamed

    def align_center(self, x):
        '''Function to align the text in the cells to the center.'''
        return ['text-align: center' for _ in x]

    def process_folder(self):
        '''Function to process the folder containing the FreezeFrame data.'''
        logging.info(f"Processing folder: {self.folder_path}")
        subfolders = [f for f in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, f))]
        logging.debug(f"Found {len(subfolders)} subfolders: {subfolders}")
        
        for subfolder in subfolders:
            logging.info(f"Processing subfolder: {subfolder}")
            self.current_ct = subfolder.split()[-2]
            logging.debug(f"Current CT: {self.current_ct}")
            try:
                self.ct_df = self.get_cohort_data(self.current_ct)
            except Exception as e:
                logging.error(f"Error getting cohort data for CT {self.current_ct}: {str(e)}")
                self.ct_df = None
            self.process_subfolder(subfolder)
            logging.info(f"Subfolder {subfolder} processed successfully!")

    def process_subfolder(self, subfolder):
        '''Function to process the FreezeFrame data for each subfolder.'''
        output_path = os.path.join(self.output, subfolder + '.xlsx')
        logging.info(f"Processing subfolder {subfolder} -> Output: {output_path}")
        writer = ExcelWriter(output_path)
        
        files = [f for f in os.listdir(os.path.join(self.folder_path, subfolder)) if f.endswith('.csv')]
        logging.debug(f"Found {len(files)} CSV files in subfolder")
        
        for file in files:
            sheet_name = file.split(os.sep)[-1].split('.')[-2]
            logging.debug(f"Processing file: {file} -> Sheet name: {sheet_name}")
            
            file_path = os.path.join(self.folder_path, subfolder, file)
            data = self.process_file(file_path, sheet_name)
            if self.ct_df is None:
                final = data
            else:
                final = pd.merge(self.ct_df, data, on='Animal ID', how='inner')
            
            if final.empty:
                logging.warning(f"No matching data found for {sheet_name}, saving original data")
                data.style.apply(self.align_center, axis=0).to_excel(writer, sheet_name=sheet_name.split('_')[-1], index=True)
                continue
                
            final.style.apply(self.align_center, axis=0).to_excel(writer, sheet_name=sheet_name.split('_')[-1], index=True)
            logging.debug(f"Sheet {sheet_name} saved with shape: {final.shape}")
            
        writer.close()
        logging.info(f"Excel file saved: {output_path}")

    def clean_columns(self, columns):
        '''Function to clean the column names.'''
        logging.debug("Cleaning column names")
        # Remove any leading or trailing whitespaces
        columns = [col.strip() for col in columns]
        # convert all columns to integers if possible
        columns = [int(float(column)) if column.replace('.', '').replace('-', '').replace('+', '').replace('e', '').isdigit() else column for column in columns]
        logging.debug(f"Cleaned columns: {columns}")
        return columns

    def process_file(self, file_path, experiment_name):
        '''Function to process the FreezeFrame data for each file.'''
        logging.info(f"Processing file: {file_path}")
        ff_df = pd.read_csv(file_path, header=1)
        logging.debug(f"File loaded with shape: {ff_df.shape}")
        
        ff_df.columns = self.clean_columns(list(ff_df.columns))
        self.experiment_name = experiment_name.split('_')[-1].lower()
        
        logging.debug(f"Processing as experiment type: {self.experiment_name}")
        if 'ltm' in self.experiment_name:
            return self.process_ltm(ff_df)
        return self.process_training(ff_df)
    
    def process_training(self, ff_df):
        '''Function to process the FreezeFrame data for the training experiment.'''
        logging.info("Processing training data")
        return self.process_data(ff_df, self.training_timestamps)
      
    def process_ltm(self, ff_df):
        '''Function to process the FreezeFrame data for the LTM experiment.'''
        logging.info("Processing LTM data")
        return self.process_data(ff_df, self.ltm_timestamps)
    
    def process_data(self, ff_df, timestamps):
        '''Function to process the FreezeFrame data for the given timestamps.'''
        logging.info(f"Processing data for experiment: {self.experiment_name}")
        df = pd.DataFrame(columns=self.get_cols(len(timestamps)))
        
        logging.debug("Extracting timestamps")
        pre_cs_start, pre_cs_end = self.extract_timestamps(timestamps, 'Pre-CS')
        cs_plus_start, cs_plus_end = self.extract_timestamps(timestamps, r'CS\+')
        iti_start, iti_end = self.extract_timestamps(timestamps, 'ITI')
        post_cs_start, post_cs_end = self.extract_timestamps(timestamps, 'Post-CS')

        animal_ids = ff_df.iloc[1:, 0].dropna().unique()
        logging.debug(f"Processing {len(animal_ids)} unique animal IDs")

        for animal_id in animal_ids:
            logging.debug(f"Processing animal ID: {animal_id}")
            pattern = f"^{re.escape(str(animal_id))}$"
            threshold = ff_df[ff_df.iloc[:, 0].astype(str).str.contains(pattern)].loc[:, 'Threshold'].values[0]

            pre_cs = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(pre_cs_start, pre_cs_end)]
            cs_plus = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(cs_plus_start, cs_plus_end)]
            iti = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(iti_start, iti_end)]
            post_cs = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(post_cs_start, post_cs_end)]
            
            valid_values = [x for x in cs_plus if x not in ('NA', None)]
            mean_cs_plus = round(np.mean(valid_values), 2) if valid_values else 'NA'

            valid_iti_values = [x for x in iti if x not in ('NA', None)]
            mean_iti = round(np.mean(valid_iti_values), 2) if valid_iti_values else 'NA'

            data = [animal_id.split()[-1], threshold, pre_cs[0], *cs_plus, mean_cs_plus, *iti, mean_iti, *post_cs]
            df = pd.concat([df, pd.DataFrame([data], columns=self.get_cols(len(timestamps)))], ignore_index=True)
            
        logging.debug(f"Processed data shape: {df.shape}")
        return df
        
    def extract_timestamps(self, timestamps, label):
        '''Function to extract the start and end timestamps for the given label.'''
        logging.debug(f"Extracting timestamps for label: {label}")
        start = timestamps[timestamps['Epoch'].str.contains(label)]['Onset'].values
        end = timestamps[timestamps['Epoch'].str.contains(label)]['Offset'].values
        logging.debug(f"Found {len(start)} timestamp pairs")
        return start, end

    def get_ff_avg(self, animal_id, start, end, ff_df):
        '''Function to calculate the average of the FreezeFrame data for the given animal ID for the given start and end timestamps.'''
        try:
            pattern = f"^{re.escape(str(animal_id))}$"
            sub_df = ff_df.loc[ff_df.iloc[:, 0].astype(str).str.contains(pattern), start:end]
            sub_df = sub_df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x).replace("NaN", 0).apply(pd.to_numeric, errors='coerce')
            avg = float(sub_df.mean(axis=1).round(2))
            logging.debug(f"Calculated average for animal {animal_id}: {avg}")
            return avg
        except Exception as e:
            if e.args[0].startswith("cannot convert the series to "):
                logging.error(f'Multiple values found for animal {animal_id}')
            elif e.args[0].startswith("cannot do slice indexing"):
                timestamp = int(e.args[0].split()[-4][1:-1])
                logging.error(f'No values for animal {animal_id} for timestamp {timestamp} in {self.experiment_name}')
            else:
                logging.error(f'Error processing animal {animal_id} from {start} to {end}: {str(e)}')
            return 'NA'
        
    def get_ct_path(self):
        '''Function to get the path to the CT file from the folder path.'''
        # walk through the folder and find the CT file which is xlsx file containing word cohort or cohorts
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.xlsx') and ('cohort' in file.lower() or 'cohorts' in file.lower()):
                    return os.path.join(root, file)
            
    def get_timestamps_path(self):
        '''Function to get the path to the timestamps file from the folder path.'''
        # check in the parent folder to find the timestamps file which is xlsx file containing word timestamps
        parent_folder = os.path.dirname(self.folder_path)
        for file in os.listdir(parent_folder):
            if file.endswith('.xlsx') and 'timestamps' in file.lower():
                return os.path.join(parent_folder, file)
        
def main():
    '''Function to parse the command line arguments and process the FreezeFrame data.'''
    args = FreezeFrame.parse_arguments()
    
    logger = logging.getLogger(__name__)
    logger.info("Starting FreezeFrame processing")
    
    output_path = args.output
    folder_path = args.folder

    logger.info(f"Command line arguments parsed successfully:\n"
                f"\tFolder path: {folder_path}\n"
                f"\tOutput path: {output_path}")

    try:
        ff = FreezeFrame(folder_path, output_path)
        logger.info("FreezeFrame object created successfully")
        
        ff.process_timestamps()
        logger.info("Timestamps processed successfully")
        
        ff.process_folder()
        logger.info("Folder processing completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}", exc_info=True)
        sys.exit(1)
        
    logger.info("Processing completed successfully")

# Run the main function
if __name__ == '__main__':
    main()