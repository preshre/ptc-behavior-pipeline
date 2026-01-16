import pandas as pd
import os
from pandas import ExcelWriter
import warnings
import sys
import re
import argparse
import logging
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
    
    def get_cols(self, num_pre_cs=3, num_iti=3, num_cs=3):
        '''Function to generate three-level MultiIndex columns with only Animal ID in basic columns.'''
        
        logging.debug(f"Getting columns for {num_pre_cs} Pre-CS, {num_iti} ITI, and {num_cs} CS")
        
        # Initialize the three levels
        top_level = []      # Main category (blank, Timestamps, % Freezing)
        mid_level = []      # Epoch type and number
        bottom_level = []   # Specific time point (-1, 0, 1)
        
        # Only Animal ID in basic columns (empty top level)
        basic_columns = ['Animal ID']
        top_level.extend([''] * len(basic_columns))
        mid_level.extend(basic_columns)
        bottom_level.extend([''] * len(basic_columns))
        
        # Timestamps section
        for phase, num in [('Pre-CS Epoch', num_pre_cs), ('ITI Epoch', num_iti), ('CS Epoch', num_cs)]:
            for i in range(1, num + 1):
                epoch_name = f"{phase} {i}"
                top_level.extend(['Timestamps'] * 3)
                mid_level.extend([epoch_name] * 3)
                bottom_level.extend(['-1', '0', '1'])
        
        # % Freezing section
        for phase, num in [('Pre-CS Epoch', num_pre_cs), ('ITI Epoch', num_iti), ('CS Epoch', num_cs)]:
            for i in range(1, num + 1):
                epoch_name = f"{phase} {i}"
                top_level.extend(['% Freezing'] * 3)
                mid_level.extend([epoch_name] * 3)
                bottom_level.extend(['-1', '0', '1'])
        
        return pd.MultiIndex.from_arrays([top_level, mid_level, bottom_level])

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
        new_df.columns = pd.MultiIndex.from_arrays([[''] + ['']*len(new_df.columns[1:]), new_df.columns, [''] + ['']*len(new_df.columns[1:])]) # set multi-level columns
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
                final = pd.merge(self.ct_df, data, how='inner')
            
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
        
        logging.debug("Extracting timestamps")
        pre_cs_start, pre_cs_end = self.extract_timestamps(timestamps, 'Pre-CS')
        cs_plus_start, cs_plus_end = self.extract_timestamps(timestamps, r'CS\+')
        iti_start, iti_end = self.extract_timestamps(timestamps, 'ITI')

        # Determine the number of epochs for each type
        num_pre_cs = len(pre_cs_start)
        num_iti = len(iti_start)
        num_cs = len(cs_plus_start)
        
        logging.debug(f"Found {num_pre_cs} Pre-CS epochs, {num_iti} ITI epochs, {num_cs} CS+ epochs")
        
        # Create dataframe with columns for all epochs
        df = pd.DataFrame(columns=self.get_cols(num_pre_cs, num_iti, num_cs))
        
        animal_ids = ff_df.iloc[1:, 0].dropna().unique()
        logging.debug(f"Processing {len(animal_ids)} unique animal IDs")

        for animal_id in animal_ids:
            logging.debug(f"Processing animal ID: {animal_id}")
            
            # Extract all transition data
            pre_cs_data = [self.get_transition_timestamps(animal_id, start, end, ff_df) 
                          for start, end in zip(pre_cs_start, pre_cs_end)]
            cs_plus_data = [self.get_transition_timestamps(animal_id, start, end, ff_df) 
                           for start, end in zip(cs_plus_start, cs_plus_end)]
            iti_data = [self.get_transition_timestamps(animal_id, start, end, ff_df) 
                       for start, end in zip(iti_start, iti_end)]
            
            # Prepare data row
            data_row = [animal_id.split()[-1]]
            
            # Add timestamp data first (first half of columns)
            for epoch_data in [pre_cs_data, iti_data, cs_plus_data]:
                for transition in epoch_data:
                    timestamps, _ = transition
                    data_row.extend(timestamps)
            
            # Then add freezing percentage data (second half of columns)
            for epoch_data in [pre_cs_data, iti_data, cs_plus_data]:
                for transition in epoch_data:
                    _, freezing_values = transition
                    data_row.extend(freezing_values)
            
            # Add the data row to the dataframe
            df = pd.concat([df, pd.DataFrame([data_row], columns=df.columns)], ignore_index=True)
            logging.debug(f"Added data row for animal {animal_id}")
            
        logging.debug(f"Processed data shape: {df.shape}")
        return df
        
    def extract_timestamps(self, timestamps, label):
        '''Function to extract the start and end timestamps for the given label.'''
        logging.debug(f"Extracting timestamps for label: {label}")
        start = timestamps[timestamps['Epoch'].str.contains(label)]['Onset'].values
        end = timestamps[timestamps['Epoch'].str.contains(label)]['Offset'].values
        logging.debug(f"Found {len(start)} timestamp pairs")
        return start, end

    def get_transition_timestamps(self, animal_id, start, end, ff_df):
        '''Function to find timestamps where freezing transitions from low (<30%) to high (>70%).
        
        Args:
            animal_id: The ID of the animal to analyze
            start: The start timestamp of the epoch to analyze
            end: The end timestamp of the epoch to analyze
            ff_df: The FreezeFrame DataFrame containing the freezing data
            
        Returns:
            A tuple containing:
            1. The actual timestamps (t, t+1, t+2) where the transition occurs
            2. The corresponding freezing percentages at these timestamps
            Returns ((NA, NA, NA), (NA, NA, NA)) if no valid transition is found.
        '''
        try:
            logging.debug(f"Finding transition timestamps for animal {animal_id} between {start} and {end}")
            
            # Convert start and end to integers
            start_int = int(start) if not isinstance(start, int) else start
            end_int = int(end) if not isinstance(end, int) else end
            
            # Ensure we have at least 3 seconds to analyze
            if end_int - start_int < 3:
                logging.warning(f"Epoch too short for animal {animal_id}: {end_int - start_int}s")
                return (('NA', 'NA', 'NA'), ('NA', 'NA', 'NA'))
            
            # Get all timestamps in the epoch
            time_range = range(start_int, end_int + 1)
            
            # Create a dictionary to store freezing percentages for each timestamp
            freezing_by_timestamp = {}
            
            # Get exact freezing % for each timestamp
            for t in time_range:
                try:
                    freezing_pct = self.get_exact_freezing_value(animal_id, t, ff_df)
                    freezing_by_timestamp[t] = freezing_pct
                except Exception as e:
                    logging.error(f"Error getting freezing value for animal {animal_id} at timestamp {t}: {str(e)}")
                    freezing_by_timestamp[t] = 'NA'
            
            # Look for transition points where freezing goes from <30% to >70%
            for t in range(start_int, end_int - 1):
                # Skip if t or t+1 is NA or not numeric
                if (freezing_by_timestamp.get(t, 'NA') == 'NA' or 
                    freezing_by_timestamp.get(t+1, 'NA') == 'NA'):
                    continue
                
                # Try to convert to float for comparison
                try:
                    pct_t = float(freezing_by_timestamp[t])
                    pct_t_plus1 = float(freezing_by_timestamp[t+1])
                except (ValueError, TypeError):
                    continue
                
                # Check if the transition meets our criteria
                if pct_t < 30 and pct_t_plus1 > 70:
                    # Get the exact freezing % at t-1, t, and t+1 (relative to the transition point)
                    fr_minus1 = freezing_by_timestamp.get(t, 'NA')
                    fr_0 = freezing_by_timestamp.get(t+1, 'NA')
                    fr_plus1 = freezing_by_timestamp.get(t+2, 'NA')
                    
                    # Validate that the transition follows the required pattern
                    if (fr_minus1 != 'NA' and fr_0 != 'NA' and 
                        float(fr_minus1) < 30 and float(fr_0) > 70 and
                        (fr_plus1 == 'NA' or 
                         (float(fr_plus1) >= 30 and float(fr_plus1) <= 100))):
                        
                        # Return both the actual timestamps and their exact freezing percentages
                        actual_timestamps = (t, t+1, t+2)
                        freezing_values = (fr_minus1, fr_0, fr_plus1)
                        
                        logging.debug(f"Found transition for {animal_id} at timestamps {actual_timestamps} with freezing values {freezing_values}")
                        return (actual_timestamps, freezing_values)
            
            logging.debug(f"No valid transition found for animal {animal_id} between {start} and {end}")
            return (('NA', 'NA', 'NA'), ('NA', 'NA', 'NA'))
            
        except Exception as e:
            logging.error(f"Error processing transition for animal {animal_id}: {str(e)}")
            return (('NA', 'NA', 'NA'), ('NA', 'NA', 'NA'))
        
    def get_exact_freezing_value(self, animal_id, timestamp, ff_df):
        '''Function to get the exact freezing value for a given animal at a specific timestamp.'''
        try:
            pattern = f"^{re.escape(str(animal_id))}$"
            # Get the exact value at the timestamp column for the animal
            sub_df = ff_df.loc[ff_df.iloc[:, 0].astype(str).str.contains(pattern), timestamp]
            
            # Clean and convert to numeric
            if isinstance(sub_df, pd.Series):
                if sub_df.dtype == 'object':
                    sub_df = sub_df.str.strip()
                freezing_value = pd.to_numeric(sub_df.replace("NaN", 0), errors='coerce')
                
                if isinstance(freezing_value, pd.Series):
                    freezing_value = freezing_value.iloc[0]
                    
                logging.debug(f"Exact freezing value for animal {animal_id} at timestamp {timestamp}: {freezing_value}")
                return round(float(freezing_value), 2)
            return 'NA'
        except Exception as e:
            if hasattr(e, 'args') and len(e.args) > 0:
                if str(e.args[0]).startswith("cannot convert the series to "):
                    logging.error(f'Multiple values found for animal {animal_id}')
                elif str(e.args[0]).startswith("cannot do slice indexing"):
                    logging.error(f'No values for animal {animal_id} for timestamp {timestamp} in {self.experiment_name}')
                else:
                    logging.error(f'Error getting freezing value for animal {animal_id} at timestamp {timestamp}: {str(e)}')
            else:
                logging.error(f'Unknown error for animal {animal_id} at timestamp {timestamp}: {str(e)}')
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