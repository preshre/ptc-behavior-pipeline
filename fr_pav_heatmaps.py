import pandas as pd
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any
import traceback
import argparse
from heatmaps_utils import ExperimentDataExtractor, ExperimentVisualizer, ExperimentBase, FreezeFrameDataProcessor

# Suppress matplotlib debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)

class PTCDataExtractor(ExperimentDataExtractor):
    """Data extractor for PTC experiment files."""
    
    def extract_experiment_timestamps(self, timestamps_path: str) -> Dict[str, Any]:
        """
        Extracts timestamps from the experiment file.

        Args:
            timestamps_path: Path to the timestamps file.

        Returns:
            Dictionary containing timestamp information.
        """
        if not timestamps_path or not os.path.exists(timestamps_path):
            logging.error(f"File not found: {timestamps_path}")
            return {}

        try:
            df = pd.read_excel(timestamps_path, header=None)

            # Initialize dictionary
            result = {}

            # Variables to track sections
            current_phase = None
            timestamps = None

            for i in range(len(df)):
                row = df.iloc[i]
                
                # Detect Training or LTM section
                if pd.notna(row[0]) and row[0] in ["Training", "LTM"]:
                    current_phase = row[0].lower()
                    result[current_phase] = {}
                
                # Detect timestamps row
                elif pd.notna(row[1]) and row[1] == "Timestamps (s)":
                    timestamps = [int(x) for x in df.iloc[i+1, 1:].tolist()]
                
                # Extract CS values
                elif pd.notna(row[0]) and row[0].startswith("CS"):
                    cs_key = row[0]
                    result[current_phase][cs_key] = {timestamps[j]: int(row[j+1]) for j in range(len(timestamps))}

            return result
        except Exception as e:
            logging.error(f"Error reading timestamps file: {e}")

        return {}

class PTCStaggeredDataExtractor(PTCDataExtractor):
    """Data extractor for PTC Staggered experiment files with delayed start timestamps."""
    
    def __init__(self, experiment_path):
        """Initialize the staggered data extractor."""
        super().__init__()
        self.staggered_timestamps = {}
        self.folder_path = experiment_path
        self.process_all_staggered_timestamps()
        
    def process_staggered_timestamps(self) -> None:
        """
        Reads and processes staggered timestamp data from an Excel file.
        
        This method uses the `self.staggered_timestamps_path` attribute, which should
        be set to the path of the staggered timestamps file before calling this method.
        """
        try:
            logging.info(f"Processing staggered timestamp file: {self.staggered_timestamps_path}")
            
            # Extract CT value from filename (similar to original implementation)
            ct = (
                os.path.basename(self.staggered_timestamps_path)
                .split(".xlsx")[0]
                .split(" ")[-3]
            )
            logging.debug(f"Extracted CT value: {ct}")

            if ct not in self.staggered_timestamps:
                self.staggered_timestamps[ct] = {}

            df = pd.read_excel(
                self.staggered_timestamps_path,
                usecols=[0, 1, 2],
                names=["Experiment", "Animal ID", "Start Delay"],
            )
            logging.debug(f"Loaded staggered timestamps with shape: {df.shape}")

            # Filter rows and drop rows with NaN values
            df = df[~df["Experiment"].str.contains("SN", na=False)].dropna(how="all")
            
            current_key = None
            for row in df.itertuples(index=False):
                experiment, animal_id, delay = row

                if isinstance(experiment, str):
                    current_key = experiment.lower()
                    if current_key not in self.staggered_timestamps[ct]:
                        self.staggered_timestamps[ct][current_key] = {}
                elif current_key and not pd.isna(animal_id) and not pd.isna(delay):
                    self.staggered_timestamps[ct][current_key][animal_id] = delay
                    logging.debug(f"Added delay {delay} for animal {animal_id}")
                    
            logging.info(f"Successfully processed staggered timestamps for CT: {ct}")
            
        except Exception as e:
            logging.error(f"Error processing staggered timestamps: {str(e)}")
            traceback.print_exc()
            
    def process_all_staggered_timestamps(self) -> None:
        """
        Find the staggered timestamps file in the experiment directory.
        
        This method searches for staggered timestamp files within the directory
        specified by the `self.folder_path` attribute, which is set during the
        initialization of the `PTCStaggeredDataExtractor` class.
        
        Returns:
            None
        """
        logging.info(f"Searching for staggered timestamps in: {self.folder_path}")
        timestamp_files_found = False

        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if 'start_timestamps' in file.lower():
                    timestamp_files_found = True
                    self.staggered_timestamps_path = os.path.join(root, file)
                    logging.debug(f"Found timestamp file: {self.staggered_timestamps_path}")
                    self.process_staggered_timestamps()
        
        if not timestamp_files_found:
            logging.warning("No timestamp files found in the specified directory")
    
class DTCDataExtractor(ExperimentDataExtractor):
    """Data extractor for DTC experiment files."""
    
    def extract_experiment_timestamps(self, timestamps_path: str) -> Dict[str, Any]:
        """
        Extracts timestamps from the DTC experiment file.

        Args:
            timestamps_path: Path to the timestamps file.

        Returns:
            Dictionary containing timestamp information.
        """
        if not timestamps_path or not os.path.exists(timestamps_path):
            logging.error(f"File not found: {timestamps_path}")
            return {}

        try:
            df = pd.read_excel(timestamps_path, header=None)

            # Initialize dictionary
            result = {}

            # Variables to track sections
            current_phase = None
            timestamps = None
            cs_type = None

            for i in range(len(df)):
                row = df.iloc[i]
                
                # Detect Training or LTM section
                if pd.notna(row[0]) and row[0] in ["Training", "LTM"]:
                    current_phase = row[0].lower()
                    result[current_phase] = {}
                    result[current_phase]["cs_plus"] = {}
                    result[current_phase]["cs_minus"] = {}
                
                # Detect timestamps row
                elif pd.notna(row[1]) and row[1] == "Timestamps (s)":
                    timestamps = [int(x) for x in df.iloc[i+1, 1:].tolist()]
                
                # Extract CS values
                elif pd.notna(row[0]):
                    cs_key = row[0]
                    
                    # Check if the row contains CS+ or CS-
                    if "+" in cs_key:
                        cs_type = "cs_plus"
                    elif "-" in cs_key:
                        cs_type = "cs_minus"
                    
                    if cs_type and cs_key.startswith("CS") and current_phase:
                        result[current_phase][cs_type][cs_key] = {timestamps[j]: int(row[j+1]) for j in range(len(timestamps))}
            return result
        except Exception as e:
            logging.error(f"Error reading timestamps file: {e}")

        return {}

class PTERDataExtractor(ExperimentDataExtractor):
    """Data extractor for PTER experiment files."""
    
    def extract_experiment_timestamps(self, timestamps_path: str) -> Dict[str, Any]:
        """
        Extracts timestamps from the PTER experiment file.

        Args:
            timestamps_path: Path to the timestamps file.

        Returns:
            Dictionary containing timestamp information.
        """
        if not timestamps_path or not os.path.exists(timestamps_path):
            logging.error(f"File not found: {timestamps_path}")
            return {}

        try:
            df = pd.read_excel(timestamps_path, header=None)

            # Initialize dictionary
            result = {}

            # Variables to track sections
            current_phase = None
            timestamps = None

            for i in range(len(df)):
                row = df.iloc[i]
                
                # Detect Training or LTM section
                if pd.notna(row[0]) and row[0] in ["PTE-1_PreITI", "PTE-1_PostITI", "PTE-2", "USRein", "LTM1d_CtxB", "LTM1d_CtxD", "LTM1d_CtxA", "LTM28d_CtxB", "LTM28d_CtxD", "LTM28d_CtxA", "Training"]:
                    current_phase = row[0].lower()
                    result[current_phase] = {}
                
                # Detect timestamps row
                elif pd.notna(row[1]) and row[0] == "Epoch":
                    timestamps = [int(x) for x in df.iloc[i+1, 1:5].dropna().tolist()]
                
                # Extract CS values
                elif pd.notna(row[0]) and row[0].startswith(("CS", "US")):
                    cs_key = row[0]
                    result[current_phase][cs_key] = {timestamps[j]: int(row[j+1]) for j in range(len(timestamps))}

            return result
        except Exception as e:
            logging.error(f"Error reading timestamps file: {e}")

        return {}

class PTCHeatmapVisualizer(ExperimentVisualizer):
    """Visualizer for heatmap plots."""
    
    def plot_data(self, all_epoch_data: Dict[str, np.ndarray], animal_id: str, group: str, 
                  output_dir: str, experiment_type: str, timestamps: Dict[str, Any]) -> None:
        """
        Plots and saves the heatmaps for the given data.

        Args:
            all_epoch_data: A dictionary of epoch data.
            animal_id: The animal ID.
            group: The animal group.
            output_dir: The output directory to save the plots.
            experiment_type: The experiment type.
            timestamps: A dictionary of timestamps.
        """
        if not all_epoch_data:
            logging.warning("No data available for heatmap plotting.")
            return

        heatmap_data = np.array(list(all_epoch_data.values())).squeeze()

        # Determine x-axis ticks
        num_timepoints = heatmap_data.shape[1]  # Total number of time bins
        time_range = np.linspace(-30, 60, num=num_timepoints)  # Mapping indices to real time
        xtick_positions = [-30, 0, 30, 60]  # Desired tick labels
        xtick_indices = [np.argmin(np.abs(time_range - t)) for t in xtick_positions]  # Find closest indices

        # Define custom colormap
        colors = ["#f7fbff", "#779ECB"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_blues", colors)

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, cmap=custom_cmap, yticklabels=list(all_epoch_data.keys()))

        # Set x-axis ticks
        plt.xticks(xtick_indices, xtick_positions)

        # Labels
        plt.ylabel("Epoch")
        plt.xlabel("Time (s)")
        plt.title(f"Animal ID - {animal_id} | Group - {group} | Experiment - {experiment_type}")

        output_path = os.path.join(output_dir, f"{experiment_type}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        # Shorten path to last 6 directories
        short_path = os.path.sep.join(output_path.split(os.path.sep)[-5:])

        logging.info(f"Saved heatmap to '.../{short_path}'")

        plt.close()

class DTCHeatmapVisualizer(ExperimentVisualizer):
    def plot_data(self, all_epoch_data: Dict[str, Dict[str, np.ndarray]], animal_id: str, group: str, 
                    output_dir: str, experiment_type: str, timestamps: Dict[str, Any]) -> None:
            """
            Plots and saves the heatmaps for CS+ and CS- data with different color schemes.

            Args:
                all_epoch_data: A dictionary with cs_plus and cs_minus data.
                animal_id: The animal ID.
                group: The animal group.
                output_dir: The output directory to save the plots.
                experiment_type: The experiment type.
                timestamps: A dictionary of timestamps.
            """
            if not all_epoch_data:
                logging.warning("No data available for heatmap plotting.")
                return
            
            # Create separate plots for CS+ and CS-
            cs_types = {
                "cs_plus": {"color": ["#FFFFFF", "#F6C7B3"], "title": "CS+"},
                "cs_minus": {"color": ["#FFFFFF", "#C3DEDD"], "title": "CS-"}
            }
            
            for cs_type, settings in cs_types.items():
                if cs_type not in all_epoch_data or not all_epoch_data[cs_type]:
                    logging.warning(f"No {cs_type} data available for animal {animal_id}.")
                    continue
                
                heatmap_data = np.array(list(all_epoch_data[cs_type].values())).squeeze()
                
                # Determine x-axis ticks
                num_timepoints = heatmap_data.shape[1]  # Total number of time bins
                time_range = np.linspace(-30, 60, num=num_timepoints)  # Mapping indices to real time
                xtick_positions = [-30, 0, 30, 60]  # Desired tick labels
                xtick_indices = [np.argmin(np.abs(time_range - t)) for t in xtick_positions]  # Find closest indices
                
                # Define custom colormap
                custom_cmap = LinearSegmentedColormap.from_list(f"custom_{cs_type}", settings["color"])
                
                # Plot heatmap
                plt.figure(figsize=(8, 6))
                sns.heatmap(heatmap_data, cmap=custom_cmap, yticklabels=list(all_epoch_data[cs_type].keys()))
                
                # Set x-axis ticks
                plt.xticks(xtick_indices, xtick_positions)
                
                # Labels
                plt.ylabel("Epoch")
                plt.xlabel("Time (s)")
                plt.title(f"Animal ID - {animal_id} | Group - {group} | {experiment_type} - {settings['title']}")
                
                output_path = os.path.join(output_dir, f"{experiment_type}_{cs_type}.png")
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                
                # Shorten path to last 6 directories
                short_path = os.path.sep.join(output_path.split(os.path.sep)[-5:])
                
                logging.info(f"Saved {cs_type} heatmap to '.../{short_path}'")
                
                plt.close()

class PTERHeatmapVisualizer(ExperimentVisualizer):
    """Visualizer for PTER experiment heatmaps."""
    
    def plot_data(self, all_epoch_data: Dict[str, np.ndarray], animal_id: str, group: str, 
                  output_dir: str, experiment_type: str, timestamps: Dict[str, Any]) -> None:
        """
        Plots and saves the heatmaps for the given data.

        Args:
            all_epoch_data: A dictionary of epoch data.
            animal_id: The animal ID.
            group: The animal group.
            output_dir: The output directory to save the plots.
            experiment_type: The experiment type.
            timestamps: A dictionary of timestamps.
        """
        if not all_epoch_data:
            logging.warning("No data available for heatmap plotting.")
            return

        heatmap_data = np.atleast_2d(np.array(list(all_epoch_data.values())).squeeze())

        # Determine x-axis ticks
        num_timepoints = heatmap_data.shape[1]  # Total number of time bins
        time_range = np.linspace(-30, 60, num=num_timepoints)  # Mapping indices to real time
        xtick_positions = [-30, 0, 30, 60]  # Desired tick labels
        xtick_indices = [np.argmin(np.abs(time_range - t)) for t in xtick_positions]  # Find closest indices

        # Define custom colormap
        colors = ["#f7fbff", "#779ECB"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_blues", colors)

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, cmap=custom_cmap, yticklabels=list(all_epoch_data.keys()))

        # Set x-axis ticks
        plt.xticks(xtick_indices, xtick_positions)

        # Labels
        plt.ylabel("Epoch")
        plt.xlabel("Time (s)")
        plt.title(f"Animal ID - {animal_id} | Group - {group} | Experiment - {experiment_type}")

        output_path = os.path.join(output_dir, f"{experiment_type}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        # Shorten path to last 6 directories
        short_path = os.path.sep.join(output_path.split(os.path.sep)[-5:])

        logging.info(f"Saved heatmap to '.../{short_path}'")

        plt.close()

class PTCFreezeFrameDataProcessor(FreezeFrameDataProcessor):
    pass

class DTCFreezeFrameDataProcessor(FreezeFrameDataProcessor):
    """Processor for DTC experiment data."""
    def _process_single_file(self, file_path: str, animal_groups: Dict[str, str], 
                            animal_output_dirs: Dict[str, str], experiment_timestamps: Dict[str, Any]) -> None:
        """
        Process a single data file with separate CS+ and CS- handling.
        
        Args:
            file_path: Path to the data file.
            animal_groups: Dictionary mapping animal IDs to groups.
            animal_output_dirs: Dictionary mapping animal IDs to output directories.
            experiment_timestamps: Dictionary containing timestamp information.
        """
        try:
            data = self._remove_empty_rows(file_path)
            
            try:
                data.columns = ['Animal ID'] + [int(float(i)) for i in data.iloc[0, 1:]]
            except ValueError:
                logging.warning("Could not convert column names to integers, using floats instead")
                try:
                    data.columns = ['Animal ID'] + [float(i) for i in data.iloc[0, 1:]]
                except Exception as e:
                    logging.error(f"Error converting column names: {e}")
                    return

            # Extract experiment type from file name
            experiment_type = os.path.basename(file_path).split("_")[1].split(".")[0]

            data = data.iloc[2:].reset_index(drop=True)
            
            n_animals = data.shape[0]
            logging.info(f"Processing {n_animals} animals")
            
            # Check if any key of experiment_timestamps is in the experiment_type
            experiment = None
            for key in experiment_timestamps.keys():
                if key in experiment_type.lower():
                    experiment = key
                    break
            
            timestamps = experiment_timestamps[experiment] if experiment else None

            for _, row in data.iterrows():
                animal_id = row["Animal ID"]
                if animal_id not in animal_groups:
                    logging.warning(f"Animal ID '{animal_id}' not found in cohorts.")
                    continue

                group = animal_groups[animal_id]
                animal_output_dir = animal_output_dirs[animal_id]

                if not timestamps:
                    logging.warning(f"No timestamps found for experiment '{experiment}'. Skipping animal '{animal_id}'.")
                    continue

                # Create separate dictionaries for CS+ and CS-
                all_epoch_data = {"cs_plus": {}, "cs_minus": {}}
                
                # Process CS+ data
                if "cs_plus" in timestamps:
                    for cs_key, values in timestamps["cs_plus"].items():
                        min_time = min(values.values())
                        max_time = max(values.values())
                        epoch_data = row.iloc[min_time:max_time+1]
                        epoch_data = epoch_data.astype(float).values.reshape(1, -1)
                        all_epoch_data["cs_plus"][cs_key] = epoch_data
                
                # Process CS- data
                if "cs_minus" in timestamps:
                    for cs_key, values in timestamps["cs_minus"].items():
                        min_time = min(values.values())
                        max_time = max(values.values())
                        epoch_data = row.iloc[min_time:max_time+1]
                        epoch_data = epoch_data.astype(float).values.reshape(1, -1)
                        all_epoch_data["cs_minus"][cs_key] = epoch_data

                self.visualizer.plot_data(all_epoch_data, animal_id, group, animal_output_dir, 
                                         experiment_type, timestamps)
        except Exception as e:
            logging.error(f"Error processing file data: {e}")
            traceback.print_exc()

class PTERFreezeFrameDataProcessor(FreezeFrameDataProcessor):
    pass

class PTCExperiment(ExperimentBase):
    """PTC experiment processor."""
    
    def __init__(self, experiment_path: str, output_base_path: str, timestamps_path: str):
        """
        Initialize PTC experiment.
        
        Args:
            experiment_path: Path to the experiment data.
            output_base_path: Base path for output.
            timestamps_path: Path to the timestamps file.
        """
        super().__init__(experiment_path, output_base_path, timestamps_path)
        
        # Initialize data processor
        self.data_extractor = PTCDataExtractor()
        self.visualizer = PTCHeatmapVisualizer()
        self.processor = PTCFreezeFrameDataProcessor(self.data_extractor, self.visualizer)

class PTCStaggeredExperiment(ExperimentBase):
    """PTC Staggered experiment processor with delayed start timestamps."""
    
    def __init__(self, experiment_path: str, output_base_path: str, timestamps_path: str):
        """
        Initialize PTC Staggered experiment.
        
        Args:
            experiment_path: Path to the experiment data.
            output_base_path: Base path for output.
            timestamps_path: Path to the timestamps file.
        """
        super().__init__(experiment_path, output_base_path, timestamps_path)
        
        # Initialize staggered data processor
        self.data_extractor = PTCStaggeredDataExtractor(experiment_path)
        self.visualizer = PTCHeatmapVisualizer()
        self.processor = PTCStaggeredFreezeFrameDataProcessor(self.data_extractor, self.visualizer)

class PTCStaggeredFreezeFrameDataProcessor(FreezeFrameDataProcessor):
    """Processor for PTC Staggered experiment data with adjusted timestamps."""
    
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
                return
            else:
                timestamps = experiment_timestamps[experiment]

            path_parts = file_path.split(os.path.sep)
            ct = None
            for part in path_parts:
                if "CT" in part:
                    ct = part
                    break
                    
            if not ct:
                logging.warning(f"Could not extract CT value from {file_path}. Using regular timestamps.")

            staggered_timestamps = getattr(self.data_extractor, 'staggered_timestamps', {})
            experiment_name = os.path.basename(os.path.dirname(file_path)).lower()

            for _, row in data.iterrows():
                animal_id = row["Animal ID"]
                delay = 0

                if animal_id not in animal_groups:
                    logging.warning(f"Animal ID '{animal_id}' not found in cohorts.")
                    continue

                group = animal_groups[animal_id]
                animal_output_dir = animal_output_dirs[animal_id]

                if ct and ct in staggered_timestamps and experiment_name in staggered_timestamps[ct]:
                    if animal_id in staggered_timestamps[ct][experiment_name]:
                        delay = staggered_timestamps[ct][experiment_name][animal_id]
                        logging.debug(f"Using start delay of {delay}s for animal {animal_id}")
                    else:
                        logging.warning(f"No delay found for animal {animal_id} in {experiment_name}")

                all_epoch_data = {}

                for key, values in timestamps.items():
                    min_time = min(values.values())
                    max_time = max(values.values())

                    # Adjust times by delay if non-zero
                    adjusted_min = max(0, min_time + delay) if delay != 0 else min_time
                    adjusted_max = min(len(row) - 1, max_time + delay) if delay != 0 else max_time

                    epoch_data = row.iloc[adjusted_min:adjusted_max+1]
                    epoch_data = epoch_data.astype(float).values.reshape(1, -1)
                    all_epoch_data[key] = epoch_data

                self.visualizer.plot_data(all_epoch_data, animal_id, group, animal_output_dir, 
                                         experiment_type, timestamps)
        except Exception as e:
            logging.error(f"Error processing file data: {e}")
            traceback.print_exc()

class DTCExperiment(ExperimentBase):
    """DTC experiment processor with CS+/CS- separation."""
    
    def __init__(self, experiment_path: str, output_base_path: str, timestamps_path: str):
        """
        Initialize DTC experiment.
        
        Args:
            experiment_path: Path to the experiment data.
            output_base_path: Base path for output.
            timestamps_path: Path to the timestamps file.
        """
        super().__init__(experiment_path, output_base_path, timestamps_path)

        # Initialize data processor
        self.data_extractor = DTCDataExtractor()
        self.visualizer = DTCHeatmapVisualizer()
        self.processor = DTCFreezeFrameDataProcessor(self.data_extractor, self.visualizer)

class PTERExperiment(ExperimentBase):
    """PTER experiment processor."""
    
    def __init__(self, experiment_path: str, output_base_path: str, timestamps_path: str):
        """
        Initialize PTER experiment.
        
        Args:
            experiment_path: Path to the experiment data.
            output_base_path: Base path for output.
            timestamps_path: Path to the timestamps file.
        """
        super().__init__(experiment_path, output_base_path, timestamps_path)
        
        # Initialize data processor
        self.data_extractor = PTERDataExtractor()
        self.visualizer = PTERHeatmapVisualizer()
        self.processor = PTERFreezeFrameDataProcessor(self.data_extractor, self.visualizer)

class ExperimentFactory:
    """Factory for creating experiment processors."""
    
    @staticmethod
    def create_experiment(experiment_type: str, experiment_path: str, output_base_path: str, 
                         timestamps_path: str) -> ExperimentBase:
        """
        Create an experiment processor of the specified type.
        
        Args:
            experiment_type: Type of experiment (PTC, DTC, PTER, PTCS).
            experiment_path: Path to the experiment data.
            output_base_path: Base path for output.
            timestamps_path: Path to the timestamps file.
            
        Returns:
            Experiment processor instance.
        """
        # Check for staggered timestamps file first
        has_staggered_timestamps = False
        for root, _, files in os.walk(experiment_path):
            if any('start_timestamps' in f.lower() for f in files):
                has_staggered_timestamps = True
                break
                
        if experiment_type.upper() == "PTC":
            if has_staggered_timestamps:
                logging.info("Detected staggered timestamps file, using PTC Staggered processor")
                return PTCStaggeredExperiment(experiment_path, output_base_path, timestamps_path)
            return PTCExperiment(experiment_path, output_base_path, timestamps_path)
        elif experiment_type.upper() == "PTCS":
            return PTCStaggeredExperiment(experiment_path, output_base_path, timestamps_path)
        elif experiment_type.upper() == "DTC":
            return DTCExperiment(experiment_path, output_base_path, timestamps_path)
        elif experiment_type.upper() == "PTER":
            return PTERExperiment(experiment_path, output_base_path, timestamps_path)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

def process_multiple_experiments(base_path: str, output_base_path: str, 
                               timestamps_path: str, experiment_type: str) -> None:
    """Process multiple experiments from subfolders."""
    logging.info(f"Processing multiple experiments in: {base_path}")
    
    # Count directories for reporting
    valid_dirs = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d)) and 'archive' not in d.lower()]
    
    logging.info(f"Found {len(valid_dirs)} directories to process")
    
    success_count = 0
    error_count = 0
    
    for dir_name in valid_dirs:
        full_path = os.path.join(base_path, dir_name)
        experiment_output_path = os.path.join(output_base_path, dir_name)
        
        # Create experiment-specific output directory if it doesn't exist
        os.makedirs(experiment_output_path, exist_ok=True)
        
        logging.info(f"Processing directory: {dir_name}")
        
        try:
            experiment = ExperimentFactory.create_experiment(
                experiment_type, full_path, experiment_output_path, timestamps_path)
            experiment.run()
            experiment.cleanup()
            success_count += 1
            logging.info(f"Successfully processed: {dir_name}")
        except Exception as e:
            error_count += 1
            logging.error(f"Error processing directory {dir_name}: {str(e)}")
    
    logging.info(f"Processing complete. Success: {success_count}, Errors: {error_count}")

def process_single_experiment(experiment_path: str, output_base_path: str, 
                             timestamps_path: str, experiment_type: str) -> None:
    """Process a single experiment directory."""
    logging.info(f"Processing single experiment at: {experiment_path}")
    try:
        experiment = ExperimentFactory.create_experiment(experiment_type, experiment_path, 
                                                        output_base_path, timestamps_path)
        experiment.run()
        experiment.cleanup()
        logging.info("Processing completed successfully")
    except Exception as e:
        logging.error(f"Error processing experiment: {str(e)}")

def main():
    """Main function to run the experiment processing."""
    parser = argparse.ArgumentParser(description="Process experiment data")
    parser.add_argument("--folder", type=str, help="Path to experiment data folder", required=True)
    parser.add_argument("--output", type=str, help="Path to output directory", required=True)
    parser.add_argument("--timestamps", type=str, help="Path to timestamps Excel file", required=True)
    parser.add_argument("--single", action="store_true", help="Process single experiment")
    parser.add_argument("--experiment-type", type=str, default="PTER", help="Experiment type (default: PTER)")
    args = parser.parse_args()

    experiment_path = os.path.expanduser(args.folder)
    output_base_path = os.path.expanduser(args.output)
    timestamps_path = os.path.expanduser(args.timestamps)
    
    if args.single:
        process_single_experiment(experiment_path, output_base_path, timestamps_path, args.experiment_type)
    else:
        process_multiple_experiments(experiment_path, output_base_path, timestamps_path, args.experiment_type)

if __name__ == "__main__":
    main()