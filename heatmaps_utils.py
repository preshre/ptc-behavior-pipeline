import pandas as pd
import os
import sys
import traceback
import chardet
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from utils import setup_logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)

class ExperimentDataExtractor(ABC):
    """Abstract base class for extracting experiment data."""
    
    def extract_animal_groups(self, file_path: str) -> Dict[str, str]:
        """
        Reads an Excel file and extracts a dictionary mapping Animal ID to Group.

        Args:
            file_path: Path to the Excel file.

        Returns:
            Dictionary with Animal ID as keys and Group as values.
        """
        animal_group_dict = {}

        if not file_path or not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return animal_group_dict

        try:
            # Load all sheets
            xls = pd.ExcelFile(file_path)

            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name, usecols=[1, 4]).dropna(how="all")

                    # Remove rows containing "Animal" in the first column
                    df = df[~df.iloc[:, 0].astype(str).str.contains("Animal", na=False, case=False)]
                    df.columns = ["Animal", "Group"]

                    # Check for required columns
                    if "Animal" not in df.columns or "Group" not in df.columns:
                        logging.warning(f"Skipping sheet '{sheet_name}': Required columns not found.")
                        continue  

                    # Drop rows with missing values
                    df = df.dropna(subset=["Animal", "Group"])

                    # strip leading/trailing whitespaces
                    df["Animal"] = df["Animal"].str.strip()
                    df["Group"] = df["Group"].str.strip()

                    # Convert to dictionary
                    animal_group_dict.update(dict(zip(df["Animal"], df["Group"])))
                except Exception as e:
                    logging.error(f"Error processing sheet '{sheet_name}': {e}")
        except Exception as e:
            logging.error(f"Error reading Excel file: {e}")

        return animal_group_dict
    
    @abstractmethod
    def extract_experiment_timestamps(self, timestamps_path: str) -> Dict[str, Any]:
        """
        Extract experiment timestamps.
        
        Args:
            timestamps_path: Path to the timestamps file.
            
        Returns:
            Dictionary containing timestamp information.
        """
        pass
    
    def find_metadata_file(self, experiment_path: str) -> Optional[str]:
        """
        Walk through the experiment path and return the path to the cohorts Excel file.

        Args:
            experiment_path: The directory to search in.

        Returns:
            Path to the first found cohorts Excel file, or None if not found.
        """
        if not os.path.exists(experiment_path):
            logging.error(f"Experiment path not found: {experiment_path}")
            return None

        for root, _, files in os.walk(experiment_path):
            for file in files:
                if file.endswith(".xlsx") and "cohort" in file.lower():
                    return os.path.join(root, file)
        
        logging.warning("No cohort Excel file found.")
        return None
    

class ExperimentVisualizer(ABC):
    """Abstract base class for experiment visualizers."""
    
    @abstractmethod
    def plot_data(self, data: Any, animal_id: str, group: str, output_dir: str, experiment_type: str, 
                  timestamps: Dict[str, Any]) -> None:
        """
        Plot experiment data.
        
        Args:
            data: The data to plot.
            animal_id: Animal ID.
            group: Animal group.
            output_dir: Directory to save the output.
            experiment_type: Type of experiment.
            timestamps: Experiment timestamps.
        """
        pass

class ExperimentDataProcessor:
    """Base class for processing experiment data."""
    
    def __init__(self, data_extractor: ExperimentDataExtractor, visualizer: ExperimentVisualizer):
        """
        Initialize with specific data extractor and visualizer.
        
        Args:
            data_extractor: Extractor for experiment data.
            visualizer: Visualizer for experiment results.
        """
        self.data_extractor = data_extractor
        self.visualizer = visualizer
    
    def process_experiment(self, experiment_path: str, output_path: str, timestamps_path: str) -> None:
        """
        Process experiment data.
        
        Args:
            experiment_path: Path to the experiment data.
            output_path: Path to save the output.
            timestamps_path: Path to the timestamps file.
        """
        pass

class ExperimentBase:
    """Base class for all experiment types."""
    
    def __init__(self, experiment_path: str, output_base_path: str, timestamps_path: str):
        """
        Initialize experiment.
        
        Args:
            experiment_path: Path to the experiment data.
            output_base_path: Base path for output.
            timestamps_path: Path to the timestamps file.
        """
        self.experiment_path = experiment_path
        self.experiment_name = os.path.basename(experiment_path)
        self.output_path = os.path.join(output_base_path, self.experiment_name)
        self.timestamps_path = timestamps_path
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Setup logging
        setup_logging(self.output_path)
        
        # Initialize data processor (to be set by subclasses)
        self.data_extractor = None
        self.visualizer = None
        self.processor = None
    
    def run(self):
        """Run the experiment processing."""
        if not self.processor:
            logging.error("Processor not initialized. Cannot run experiment.")
            return
            
        logging.info(f"Starting processing for experiment: {self.experiment_name}")
        self.processor.process_experiment(self.experiment_path, self.output_path, self.timestamps_path)
        logging.info(f"Processing completed for experiment: {self.experiment_name}")

    def cleanup(self):
        """Walk through the output directory and remove empty directories."""
        logging.info("Cleaning up empty directories...")
        for root, dirs, _ in os.walk(self.output_path, topdown=False):
            if not os.listdir(root):
                os.rmdir(root)

        logging.info("Cleanup completed.")

class FreezeFrameDataProcessor(ExperimentDataProcessor):
    """Processor for freeze data experiments like PTC."""
    
    def process_experiment(self, experiment_path: str, output_path: str, timestamps_path: str) -> None:
        """
        Process freeze data experiment.
        
        Args:
            experiment_path: Path to the experiment data.
            output_path: Path to save the output.
            timestamps_path: Path to the timestamps file.
        """
        # Get cohorts file path
        cohorts_xlsx_path = self.data_extractor.find_metadata_file(experiment_path)

        if not cohorts_xlsx_path:
            logging.error("No valid cohort file found. Exiting function.")
            return
        
        # Extract animal groups
        animal_groups = self.data_extractor.extract_animal_groups(cohorts_xlsx_path)

        if not animal_groups:
            logging.error("No animal groups extracted. Exiting function.")
            return
        
        # Extract experiment timestamps
        experiment_timestamps = self.data_extractor.extract_experiment_timestamps(timestamps_path)

        # Create output directories
        animal_output_dirs = self._create_output_directories(output_path, animal_groups)
        
        # Process data files
        self._process_data_files(experiment_path, animal_groups, animal_output_dirs, experiment_timestamps)
    
    def _create_output_directories(self, output_path: str, animal_groups: Dict[str, str]) -> Dict[str, str]:
        """
        Create output directories for each animal.
        
        Args:
            output_path: Base output path.
            animal_groups: Dictionary mapping animal IDs to groups.
            
        Returns:
            Dictionary mapping animal IDs to their output directories.
        """
        animal_output_dirs = {}
        for animal_id, group in animal_groups.items():
            animal_output_dir = os.path.join(output_path, group, animal_id)
            animal_output_dirs[animal_id] = animal_output_dir
            os.makedirs(animal_output_dir, exist_ok=True)
        return animal_output_dirs
    
    def _process_data_files(self, experiment_path: str, animal_groups: Dict[str, str], 
                           animal_output_dirs: Dict[str, str], experiment_timestamps: Dict[str, Any]) -> None:
        """
        Process data files.
        
        Args:
            experiment_path: Path to the experiment data.
            animal_groups: Dictionary mapping animal IDs to groups.
            animal_output_dirs: Dictionary mapping animal IDs to output directories.
            experiment_timestamps: Dictionary containing timestamp information.
        """
        for root, _, files in os.walk(experiment_path):
            for file in files:
                if file.endswith(".csv") and file.lower().startswith("freeze_"):
                    logging.info(f"Processing file: {file}")
                    file_path = os.path.join(root, file)

                    try:
                        self._process_single_file(file_path, animal_groups, animal_output_dirs, experiment_timestamps)
                    except KeyboardInterrupt:
                        logging.warning("Keyboard interrupt detected. Exiting function.")
                        sys.exit(0)
                    except Exception as e:
                        logging.error(f"Error processing file '{file_path}': {e}")

    def _remove_empty_rows(self, file_path: str) -> pd.DataFrame:
        """
        Remove empty rows and columns from a CSV file with robust encoding handling.

        Args:
            file_path: Path to the CSV file.

        Returns:
            DataFrame with empty rows and columns removed.
        """
        encodings_to_try = ['utf-8', 'macroman', 'ISO-8859-1']

        # Step 1: Try chardet detection
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        guess = chardet.detect(raw_data).get('encoding', None)

        # Step 2: Try guessed encoding first, then fallback options
        tried_encodings = [guess] + [enc for enc in encodings_to_try if enc != guess]

        for enc in tried_encodings:
            try:
                logging.info(f"Trying to read {file_path} with encoding: {enc}")
                data = pd.read_csv(file_path, encoding=enc)
                break
            except UnicodeDecodeError:
                logging.warning(f"Failed to decode {file_path} with encoding: {enc}")
        else:
            raise UnicodeDecodeError(f"All decoding attempts failed for file: {file_path}")

        # Step 3: Drop fully empty rows and columns
        data = data.dropna(how='all')
        data = data.dropna(axis=1)

        return data
    
    def _process_single_file(self, file_path: str, animal_groups: Dict[str, str], 
                            animal_output_dirs: Dict[str, str], experiment_timestamps: Dict[str, Any]) -> None:
        """
        Process a single data file.
        
        Args:
            file_path: Path to the data file.
            animal_groups: Dictionary mapping animal IDs to groups.
            animal_output_dirs: Dictionary mapping animal IDs to output directories.
            experiment_timestamps: Dictionary containing timestamp information.
        """
        try:
            data = self._remove_empty_rows(file_path)
            
            try:
                data.columns = ['Animal ID'] + [int(float(i)) for i in data.iloc[0, 1:] if pd.notna(i)]
            except ValueError:
                logging.warning("Could not convert column names to integers, using floats instead")
                try:
                    data.columns = ['Animal ID'] + [float(i) for i in data.iloc[0, 1:] if pd.notna(i)]
                except Exception as e:
                    logging.error(f"Error converting column names: {e}")
                    return

            # Extract experiment type from file name
            experiment_type = os.path.basename(file_path).split("freeze_")[-1].split(".")[0]

            logging.info(f"Processing experiment type: {experiment_type}")

            data = data.iloc[2:].reset_index(drop=True)
            
            n_animals = data.shape[0]
            logging.info(f"Processing {n_animals} animals")
            
            # Check if any key of experiment_timestamps is in the experiment_type
            experiment = None
            for key in experiment_timestamps.keys():
                if key in experiment_type.lower():
                    experiment = key
                    break

            if experiment not in experiment_timestamps:
                logging.warning(f"No timestamps found for experiment '{experiment}'.")
                logging.debug(f"Available timestamps: {list(experiment_timestamps.keys())}")
                return
            else:
                timestamps = experiment_timestamps[experiment]

            for _, row in data.iterrows():
                animal_id = row["Animal ID"]
                if animal_id not in animal_groups:
                    logging.warning(f"Animal ID '{animal_id}' not found in cohorts.")
                    continue

                group = animal_groups[animal_id]
                animal_output_dir = animal_output_dirs[animal_id]

                all_epoch_data = {}

                for key, values in timestamps.items():
                    min_time = min(values.values())
                    max_time = max(values.values())
                    epoch_data = row.iloc[min_time:max_time+1]
                    epoch_data = epoch_data.astype(float).values.reshape(1, -1)
                    all_epoch_data[key] = epoch_data

                self.visualizer.plot_data(all_epoch_data, animal_id, group, animal_output_dir, 
                                         experiment_type, timestamps)
        except Exception as e:
            logging.error(f"Error processing file data: {e}")
            traceback.print_exc()