import os
import logging
from datetime import datetime
import re
from pathlib import Path
import sys

# Signature for Behavior Data Analysis System
# Developed by Harshil Sanghvi for Shrestha Lab
SIGNATURE = "Developed by Harshil Sanghvi for Shrestha Lab"

def setup_logging(output_dir):
    """
    Configure logging to both file and console with different levels
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{os.path.basename(output_dir)}_{timestamp}.log')
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    
    # Add signature to log file
    logging.info(f"=== {SIGNATURE} ===")

def plot_pair(ax, xmin, xmax, ymin, ymax, alpha, color, label="", offset_bar_color='#a7151e'):
    """
    Plot a highlighted region with an offset bar above it.
    """
    logging.debug(f"Plotting pair with coords: ({xmin}, {ymin}) - ({xmax}, {ymax}), label: {label}, offset bar color: {offset_bar_color}")
    ax.plot([xmin, xmax], [ymax + 1, ymax + 1], color=offset_bar_color, linewidth=6, label=label)