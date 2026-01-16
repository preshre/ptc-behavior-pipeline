from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pandas import ExcelWriter
import warnings
import argparse
import traceback
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from utils import setup_logging
import os

warnings.filterwarnings("ignore")

@dataclass
class ProcessingStats:
    """Statistics for processing runs"""
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    error_files: int = 0
    start_time: datetime = None
    end_time: datetime = None

class FreezeFrame(ABC):
    """Abstract base class for FreezeFrame data processing"""
    
    def __init__(self):
        """Initialize FreezeFrame processor"""
        self.ct_path: Optional[Path] = None
        self.folder_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self.output: Optional[Path] = None
        self.timestamps: Optional[Dict] = None
        self.ct_df: Optional[pd.DataFrame] = None
        self.stats = ProcessingStats()
        
    @abstractmethod
    def get_cols(self, num_of_cs: int) -> pd.MultiIndex:
        """Get column names for the DataFrame"""
        pass

    @staticmethod
    def find_matching_animal(animal_id: str, timestamps: Dict, ct: str, experiment_name: str) -> Optional[str]:
        """Find matching animal key in timestamps with string comparison."""
        animal_str = str(animal_id)
        for key in timestamps[ct][experiment_name]:
            if animal_str in str(key) or str(key) in animal_str:
                return key
        return None
    
    def get_cohort_data(self, ct: str, dt: str) -> pd.DataFrame:
        """Extract cohort data from the CT file"""
        logging.info(f"Extracting cohort data for CT: {ct}, DT: {dt}")
        try:
            df = pd.read_excel(self.ct_path, usecols=range(5))
            logging.debug(f"CT file loaded, shape: {df.shape}")
            
            mask = (df['Unnamed: 0'].str.contains(ct, na=False)) & (df['Unnamed: 0'].str.contains(dt, na=False))
            ct_row_indices = df.index[mask].tolist()
            
            if not ct_row_indices:
                error_msg = f"No matching data found for CT {ct} and DT {dt}"
                logging.error(error_msg)
                raise ValueError(error_msg)
                
            ct_row_index = ct_row_indices[0]
            logging.debug(f"Found data at row index: {ct_row_index}")
            
            # Extract rows until blank row
            cohort_data = []
            for row in df.iloc[ct_row_index+1:].itertuples():
                row_values = list(row)[1:]  # Skip the index
                if all(pd.isna(value) for value in row_values):
                    break
                cohort_data.append(row)
            
            if len(cohort_data) < 2:
                error_msg = f"Insufficient data found for CT {ct}"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            # Process the cohort data
            new_df = pd.DataFrame([row[1:] for row in cohort_data[1:]], columns=cohort_data[0][1:])
            new_df.drop(new_df.columns[0], axis=1, inplace=True)
            new_df.reset_index(drop=True, inplace=True)
            
            # Set multi-level columns
            new_df.columns = pd.MultiIndex.from_arrays([
                new_df.columns,
                [''] * len(new_df.columns)
            ])
            
            result = new_df.rename(columns={'Animal': 'Animal ID'})
            logging.info(f"Cohort data processed successfully, shape: {result.shape}")
            return result
            
        except FileNotFoundError:
            logging.error(f"CT file not found: {self.ct_path}")
            raise
        except Exception as e:
            logging.error(f"Error processing cohort data for CT {ct}: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def align_center(x: pd.Series) -> List[str]:
        """Apply center alignment to cells"""
        return ['text-align: center'] * len(x)

    def process_folder(self) -> None:
        """Process folder containing FreezeFrame data"""
        logging.info(f"Starting folder processing: {self.folder_path}")
        self.stats.start_time = datetime.now()
        
        try:
            items = list(self.folder_path.iterdir())
            logging.debug(f"Found {len(items)} items in folder")
        except Exception as e:
            logging.critical(f"Failed to access folder {self.folder_path}: {str(e)}", exc_info=True)
            raise

        def extract_ct_from_name(name: str) -> Optional[str]:
            """Extract CT identifier from name"""
            ct_match = re.search(r'CT\d+', name)
            return ct_match.group() if ct_match else None

        def should_skip_folder(name: str) -> bool:
            """Determine if folder should be skipped"""
            skip_patterns = {'archive', 'timestamps', 'backup', 'old', 'temp'}
            return any(pattern in name.lower() for pattern in skip_patterns)

        def process_single_location(location_name: str, is_subfolder: bool = True) -> None:
            """Process a single location (folder or file)"""
            logging.info(f"Processing location: {location_name}")
            
            ct = extract_ct_from_name(location_name)
            if not ct and is_subfolder:
                logging.warning(f"Skipping '{location_name}': no valid CT identifier")
                self.stats.skipped_files += 1
                return

            if ct:
                try:
                    dt = Path(location_name).name.split()[0]
                    logging.info(f"Getting cohort data for {ct} and {dt}")
                    self.ct_df = self.get_cohort_data(ct, dt)
                except Exception as e:
                    logging.error(f"Failed to get cohort data for {ct}: {str(e)}")
                    self.ct_df = None

            try:
                if is_subfolder:
                    self.process_subfolder(location_name)
                else:
                    original_path = self.folder_path
                    self.folder_path = self.folder_path.parent
                    self.process_subfolder(location_name)
                    self.folder_path = original_path
            except Exception as e:
                logging.error(f"Error processing {location_name}: {str(e)}", exc_info=True)
                self.stats.error_files += 1

        # Check for Excel files
        excel_files = {f for f in items 
                      if f.suffix.lower() == '.xlsx' 
                      and f.is_file() 
                      and 'cohorts' not in f.name.lower()}

        if excel_files:
            logging.info("Processing folder with direct Excel files")
            process_single_location(self.folder_path.name, is_subfolder=False)
            return

        # Process subfolders
        subfolders = [f for f in items 
                     if f.is_dir() 
                     and not should_skip_folder(f.name)]
        
        logging.info(f"Found {len(subfolders)} subfolders to process")
        
        for subfolder in subfolders:
            logging.info(f"Processing subfolder: {subfolder.name}")
            process_single_location(subfolder.name)

        self.stats.end_time = datetime.now()
        self._log_processing_stats()

    def _log_processing_stats(self) -> None:
        """Log processing statistics"""
        duration = self.stats.end_time - self.stats.start_time
        logging.info("\n=== Processing Statistics ===")
        logging.info(f"Total files found: {self.stats.total_files}")
        logging.info(f"Successfully processed: {self.stats.processed_files}")
        logging.info(f"Skipped: {self.stats.skipped_files}")
        logging.info(f"Errors: {self.stats.error_files}")
        logging.info(f"Total processing time: {duration}")
        logging.info("==========================")

    def process_subfolder(self, subfolder: str) -> None:
        """Process FreezeFrame data in a subfolder"""
        output_path = self.output / f"{subfolder}.xlsx"
        logging.info(f"Processing subfolder {subfolder} -> {output_path}")
        
        subfolder_path = self.folder_path / subfolder
        ct = subfolder.split()[-2]
        
        with ExcelWriter(output_path) as writer:
            for file in subfolder_path.glob('*.csv'):
                self.stats.total_files += 1
                
                sheet_name = file.stem.split('_')[-1]
                if not any(keyword in sheet_name for keyword in ['SAA', 'LTM', 'training']):
                    logging.info(f"Skipping sheet '{sheet_name}': missing required keywords")
                    self.stats.skipped_files += 1
                    continue

                try:
                    logging.info(f"Processing file: {file.name}")
                    data = self.process_file(file, sheet_name, ct)
                    
                    final = data if self.ct_df is None else pd.merge(self.ct_df, data, on='Animal ID', how='inner')
                    
                    if final.shape[0] == 0:
                        final = data
                        logging.warning(f"No matching animals found for {ct} in experiment details excel")

                    final.index += 1
                    
                    final.style.apply(self.align_center, axis=0).to_excel(writer, sheet_name=sheet_name, index=True)
                    logging.info(f"Successfully processed: {file.name}")
                    self.stats.processed_files += 1
                    
                except KeyError as ke:
                    logging.error(f"Timestamp error for {ct}: {str(ke)}")
                    self.stats.error_files += 1
                except Exception as e:
                    logging.error(f"Error processing {file.name}: {str(e)}", exc_info=True)
                    self.stats.error_files += 1

    @staticmethod
    def clean_columns(columns: List[str]) -> List[Union[int, str]]:
        """Clean and standardize column names"""
        def convert_to_numeric(col: str) -> Union[int, str]:
            col = col.strip()
            try:
                return int(float(col)) if re.match(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', col) else col
            except ValueError:
                return col
        
        return [convert_to_numeric(col) for col in columns]

    def process_file(self, file_path: Path, experiment_name: str, ct: str) -> pd.DataFrame:
        """Process individual FreezeFrame data file"""
        logging.info(f"Processing file: {file_path.name}")
        try:
            ff_df = pd.read_csv(file_path, header=1)
            logging.debug(f"File loaded, shape: {ff_df.shape}")
            
            ff_df.columns = self.clean_columns(list(ff_df.columns))
            return self.process_experiment(ff_df, experiment_name, ct)
            
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            raise

    @abstractmethod
    def process_experiment(self, ff_df: pd.DataFrame, experiment_name: str, ct: str) -> pd.DataFrame:
        """Process experiment data"""
        pass

    def get_ff_avg(self, animal_id: str, start: float, end: float, ff_df: pd.DataFrame) -> Union[float, str]:
        """Calculate average FreezeFrame data"""
        logging.debug(f"Calculating average for animal {animal_id} ({start}-{end})")
        try:
            first_column_str = ff_df.iloc[:, 0].astype(str)
            matches_exact_animal_id = first_column_str == animal_id
            sub_df = ff_df.loc[matches_exact_animal_id, int(start):int(end)]
            
            # Clean and convert data
            sub_df = (sub_df.replace({"NaN": "0"})
                           .apply(pd.to_numeric, errors='coerce'))
            
            result = float(sub_df.mean(axis=1).round(2))
            logging.debug(f"Calculated average: {result}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            if "cannot convert the series to <class 'float'>" in error_msg:
                logging.error(f"Multiple rows found for {animal_id}")
                return 999
            if match := re.search(r'\[(\d+)\]', error_msg):
                timestamp = match.group(1)
                logging.error(f"Timestamp {timestamp} not found for {animal_id}")
                return 'NA'
            
            logging.error(f"Error calculating average for {animal_id}: {error_msg}")
            return 'NA'

    @abstractmethod
    def parse_sheet(self, xlsx: pd.ExcelFile, sheet: str) -> Any:
        """Parse Excel sheet"""
        pass

    def process_sheets(self) -> Dict:
        """Process all timestamp sheets"""
        logging.info("Processing timestamp sheets")
        try:
            all_data = {}

            # Walk through the directory and subdirectories to find all `.xlsx` files
            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    if file.endswith('.xlsx') and not re.search(r'\bcohorts?\b', file, re.IGNORECASE):
                        file_path = Path(root) / file

                        ct_match = re.search(r'CT\d+', file)
                        if not ct_match:
                            continue

                        ct = ct_match.group()
                        logging.info(f"Processing timestamps for {ct}")

                        xlsx = pd.ExcelFile(file_path)
                        dfs = {
                            sheet.upper().replace(' ', ''): self.parse_sheet(xlsx, sheet)
                            for sheet in xlsx.sheet_names
                        }

                        all_data[ct] = dfs
                        logging.info(f"Processed timestamps for {ct}")

            logging.info("Processed all timestamps successfully")
            return all_data

        except Exception as e:
            logging.error(f"Error processing timestamp sheets: {str(e)}", exc_info=True)
            raise

    def get_ct_path(self) -> Path:
        """Walk through self.folder_path to find the CT file"""
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.xlsx') and re.search(r'\bcohorts?\b', file, re.IGNORECASE):
                    return Path(root) / file
        raise FileNotFoundError("CT file not found")
        
    def parse_arguments(self) -> None:
        """Parse command line arguments and set up logging"""
        parser = argparse.ArgumentParser(description='Process FreezeFrame data')
        parser.add_argument('--folder', type=str, required=True, 
                          help='Path to the folder containing the FreezeFrame data')
        parser.add_argument('--output', type=str, required=True, 
                          help='Path to the output folder')
        
        args = parser.parse_args()
        
        # Convert all paths to Path objects
        self.folder_path = Path(args.folder)
        self.ct_path = self.get_ct_path()
        self.output_path = Path(args.output)
        self.output = self.output_path
        
        # Validate paths
        for path, name in [
            (self.ct_path, "CT file"),
            (self.folder_path, "Data folder")
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        setup_logging(self.output_path)
        
        logging.info("Arguments parsed and validated successfully")
        logging.debug(f"CT path: {self.ct_path}")
        logging.debug(f"Folder path: {self.folder_path}")
        logging.debug(f"Output path: {self.output_path}")

    def main(self) -> None:
        """Main execution function"""
        try:
            logging.info("Starting FreezeFrame processing")
            
            # Parse arguments and set up logging
            self.parse_arguments()
            
            # Process timestamp sheets
            logging.info("Processing timestamp sheets...")
            self.timestamps = self.process_sheets()
            logging.info("Timestamp processing completed")
            
            # Process data folder
            logging.info("Processing data folder...")
            self.process_folder()
            logging.info("Data folder processing completed")
            
            # Log final statistics
            duration = self.stats.end_time - self.stats.start_time
            logging.info("\n=== Final Processing Summary ===")
            logging.info(f"Total files processed: {self.stats.total_files}")
            logging.info(f"Successfully processed: {self.stats.processed_files}")
            logging.info(f"Files skipped: {self.stats.skipped_files}")
            logging.info(f"Files with errors: {self.stats.error_files}")
            logging.info(f"Total processing time: {duration}")
            logging.info(f"Average time per file: {duration / self.stats.total_files if self.stats.total_files > 0 else 0}")
            logging.info("Processing completed successfully")
            
        except KeyboardInterrupt:
            logging.critical("Processing interrupted by user")
            raise
        except Exception as e:
            logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
            raise
        finally:
            if self.stats.start_time and not self.stats.end_time:
                self.stats.end_time = datetime.now()
            logging.info("FreezeFrame processing finished")