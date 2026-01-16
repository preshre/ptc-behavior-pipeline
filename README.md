# Pavlovian Threat Conditioning (PTC) Analysis Pipeline

**FreezeFrame Data Processing and Heatmap Visualization**

Developed by Harshil Sanghvi for Shrestha Lab

---

## Overview

This automated pipeline processes behavioral neuroscience data from Pavlovian Threat Conditioning (PTC) experiments. The system handles FreezeFrame recordings and generates comprehensive freezing behavior analyses with temporal heatmap visualizations for threat-related learning in mice.

### Key Capabilities

The lab's research focuses on threat-related learning and avoidance behavior in mice, examining how organisms perceive, predict, and respond to environmental threats. PTC assesses passive defensive responses through classical fear conditioning and freezing behavior, commonly used to investigate amygdala function and fear circuitry. These paradigms provide insight into adaptive vs. maladaptive learning, fear generalization, and individual differences, contributing to understanding of PTSD, anxiety disorders, and phobias.

The scripts incorporate intelligent auto-detection mechanisms that identify necessary files based on standardized naming conventions and folder hierarchies, including automatic detection of cohort information, timestamp data, and experimental phase identification. When proper folder structures are maintained, most scripts require only --folder and --output parameters for execution. The pipeline integrates multiple data sources including FreezeFrame CSV exports, cohort Excel files, and experimental timestamps, using sophisticated data fusion algorithms to correlate temporal behavioral data with experimental events.

---

## System Requirements

### Operating Systems
- **macOS**: 10.15 (Catalina) or later
- **Linux**: Ubuntu 18.04 LTS or later, CentOS 7 or later
- **Windows**: Windows 10 or later

**Tested on:** macOS Monterey (12.x), Ubuntu 20.04 LTS, Windows 10/11

### Software Dependencies

#### Core Requirements
- **Python**: 3.8 or later

**Tested on:** Python 3.8, 3.9, 3.10, 3.11

#### Required Python Packages
```
pandas==2.2.3
numpy==2.1.2
matplotlib==3.9.3
seaborn==0.13.2
openpyxl==3.1.5
chardet==5.2.0
jinja2==3.1.6
tqdm==4.67.1
apscheduler==3.11.0
argparse==1.4.0
```

### Hardware Requirements
- **Minimum**: 
  - 4 GB RAM
  - 2 CPU cores
  - 500 MB available disk space
- **Recommended**:
  - 8 GB RAM or more
  - 4+ CPU cores
  - 2 GB available disk space

### Non-Standard Hardware
No specialized hardware required.

---

## Installation Guide

### Step 1: Python Installation

**macOS:**
```bash
# Check Python version
python3 --version

# Install Python 3.8+ using Homebrew
brew install python@3.11
```

**Windows:**
Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/), ensuring PATH variable configuration during installation.

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.11

# CentOS/RHEL
sudo yum install python3.11
```

### Step 2: Extract Pipeline Files

Extract the provided zip file to your desired location:

```bash
unzip ptc-analysis-pipeline.zip
cd ptc-analysis-pipeline
```

### Step 3: Install Dependencies

Install all required packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, matplotlib, seaborn, openpyxl; print('All dependencies installed successfully!')"
```

### Typical Installation Time

- **Package installation**: 2-5 minutes on a standard desktop computer with normal internet connection
- **Total setup time**: 5-10 minutes

---

## Demo

### Demo Dataset Location

A demonstration dataset is provided in the `demo_data/` folder containing a complete PTC experiment with:
- FreezeFrame CSV files (Training and LTM sessions)
- Cohorts Excel file with animal group assignments
- PTC Timestamps Excel file with experimental epoch timing

**Demo Data Structure:**
```
Demo_Data/
├── PTC
    └── PTC Timestamps.xlsx
    ├── WT_PTC PTC Cohorts.xlsx
    └── WT_PTC
        ├── WT_PTC PTC Cohorts.xlsx
        └── 20250908 WT_PTC CT1 PTC
            ├── Freeze_LTM28d.csv
            └── Freeze_training.csv
```

### Running the Demo

#### Demo 1: Process PTC FreezeFrame Data

Process freezing behavior data and extract epoch-specific metrics:

```bash
python ptc_ff.py --folder "Demo_Data/PTC/WT_PTC" --output "Demo_Output"
```

**Expected Output:**
- Excel file: `demo_output/20250908 WT_PTC CT1 PTC.xlsx`
- Sheets: training, LTM28d
- Processing log: `Demo_Output/logs/`

**Exact Run Time for Demo Data:** ~1.04 seconds ≈ 1 second on a standard desktop computer

#### Demo 2: Generate Freezing Response Heatmaps

Create temporal heatmap visualizations showing freezing patterns:

```bash
python fr_pav_heatmaps.py --folder "Demo_Data/PTC" --output "Demo_Output" --timestamps "Demo_Data/PTC Timestamps for Heatmaps.xlsx" --experiment-type PTC
```

**Expected Output:**
- Heatmap images organized by group and animal ID
- Directory structure to find Heatmaps: `Demo_Output/WT_PTC/WT_PTC/WT PTC | RcLTM_CtxB | RmLTM_CtxC`
- Files: `Freeze_training.png`, `Freeze_LTM28d.png` for each animal
- PNG format with color-coded freezing intensity (0-100%)

**Exact Run Time for Heatmaps generation:** ~2.57 seconds ≈ 2.6 seconds on a standard desktop computer

### Expected Output Validation

After running both demos, verify:
1. **Processed Data**: Excel file with multi-level column headers showing epoch-wise freezing data
2. **Heatmaps**: PNG images showing temporal freezing patterns with clear epoch boundaries
3. **Logs**: Detailed processing logs with no critical errors
4. **File Count**: One Excel file per cohort, multiple heatmaps per animal (one per experimental phase)
5. **Output**: In `Demo_Output/WT_PTC` folder

---

## Instructions for Use

### Input Data Requirements

Your PTC experiment data must follow this folder architecture:

```
PTC_Experiment/
├── PTC Timestamps.xlsx                        # Auto-detected timestamps file
└── ExperimentName_PTC/
    ├── ExperimentName_PTC_Cohorts.xlsx       # Auto-detected cohort file
    ├── YYYYMMDD_ExperimentName_CT1_PTC/
    │   ├── Freeze_training.csv
    │   └── Freeze_LTM1d.csv
    ├── YYYYMMDD_ExperimentName_CT2_PTC/
    │   ├── Freeze_training.csv
    │   └── Freeze_LTM1d.csv
    └── Additional_CT_Subfolders/
```

### File Format Specifications

#### PTC Timestamps File
**Format:** Excel (.xlsx)

**Structure:**
- Contains two main sections: "Training" and "LTM"
- Each section has columns: Epoch, Onset, Offset
- Epochs include: Pre-CS, CS+1, CS+2, ..., ITI1, ITI2, ..., Post-CS
- Times in seconds from experiment start

**Example:**
```
Training
Epoch       Onset    Offset
Pre-CS      0        180
CS+1        180      210
ITI1        210      330
CS+2        330      360
Post-CS     720      900
```

#### Cohorts File
**Format:** Excel (.xlsx)

**Content:**
- Column 1: Animal ID
- Column 2: Sex (optional)
- Column 3: Subject identifier (optional)
- Column 4: Group assignment (e.g., "Control", "Experimental")
- Auto-detected by filename containing "cohort"

#### FreezeFrame CSV Files
**Format:** CSV with comma separation

**Structure:**
- Row 1: Threshold settings
- Row 2: Time points (seconds) as column headers
- Row 3+: Animal ID followed by freezing percentages (0-100%) at each time point

**Naming Convention:** `Freeze_[experiment_phase].csv`

Example: `Freeze_training.csv`, `Freeze_LTM28d.csv`

### Running the Pipeline on Your Data

#### 1. Primary PTC Processing

Standard PTC FreezeFrame analysis with CS+ conditioning protocols:

```bash
python ptc_ff.py --folder "/path/to/PTC_experiment" --output "/path/to/analysis_results"
```

**What this does:**
- Auto-detects cohort and timestamp files
- Processes all FreezeFrame CSV files in CT subfolders
- Extracts freezing percentages for Pre-CS, CS+, ITI, Post-CS epochs
- Generates Excel files with multi-level column headers
- Creates processing logs in output directory

**Output:** 
- One Excel file per CT subfolder
- Sheets for each experimental phase (Training, LTM1d, etc.)
- Integrated cohort information (Animal ID, Sex, Subject, Group)


#### 3. Heatmap Visualization

Generate comprehensive temporal heatmaps showing freezing patterns:

```bash
# PTC-specific heatmap generation
python fr_pav_heatmaps.py --folder "/path/to/PTC_data" --output "/path/to/PTC_heatmaps" --timestamps "/path/to/PTC_timestamps.xlsx" --experiment-type PTC

# Single directory processing (for single CT)
python fr_pav_heatmaps.py --folder "/path/to/single_CT_folder" --output "/path/to/output" --timestamps "/path/to/timestamps.xlsx" --experiment-type PTC --single

# Multiple experiment processing (default, processes all CT subfolders)
python fr_pav_heatmaps.py --folder "/path/to/PTC_experiment" --output "/path/to/heatmaps" --timestamps "/path/to/timestamps.xlsx" --experiment-type PTC
```

**Parameters:**
- `--folder`: Path to experiment data
- `--output`: Directory for heatmap output
- `--timestamps`: Path to PTC timestamps Excel file (required)
- `--experiment-type`: Specify "PTC" for Pavlovian Threat Conditioning
- `--single`: Optional flag to process only a single experiment directory

**Heatmap Features:**
- Individual tone analysis with pre-CS, CS, and post-CS temporal windows
- Color-coded freezing intensity (blue = low freezing, red = high freezing)
- Separate heatmaps for Training and LTM phases
- Organized output by animal group and ID


### Output Directory Structure

The pipeline generates organized output:

```
output_directory/
├── 20240115_ExperimentName_CT1_PTC.xlsx      # Processed data
├── 20240115_ExperimentName_CT2_PTC.xlsx
├── logs/
│   ├── processing_20240115_143022.log
│   └── error_reports/
└── heatmaps/
    ├── Control_Group/
    │   ├── Animal_001/
    │   │   ├── training_heatmap.png
    │   │   └── ltm_heatmap.png
    │   └── Animal_002/
    └── Experimental_Group/
        └── Animal_003/
```

### Troubleshooting

**Issue**: "Timestamps file not found"
- **Solution**: Ensure PTC Timestamps.xlsx is in the parent directory of your experiment folder
- File must contain "timestamp" in filename (case-insensitive)

**Issue**: "CT file not found"
- **Solution**: Verify cohorts.xlsx exists within the experiment folder
- File must contain "cohort" or "cohorts" in filename (case-insensitive)

**Issue**: "No matching data found for CT"
- **Solution**: Check that animal IDs in FreezeFrame CSV files match those in cohorts.xlsx
- Ensure CT identifier in folder name matches cohorts file

**Issue**: Empty Excel output files
- **Solution**: Verify timestamps match the time range in FreezeFrame CSV files
- Check that CSV files have proper format (Animal ID in first column)
- Review processing logs in output/logs/ directory

**Issue**: Missing heatmaps for some animals
- **Solution**: Confirm animal ID exists in both cohorts file and FreezeFrame CSV
- Verify timestamps file has correct epoch definitions
- Check logs for specific error messages about missing animals

**Issue**: "Multiple values found for animal"
- **Solution**: Ensure each animal ID appears only once per time point in FreezeFrame CSV
- Check for duplicate rows with the same animal ID

---

## File Descriptions

### Core Processing Scripts

**`ptc_ff.py`**
- Primary PTC FreezeFrame data processor
- Processes CS+ conditioning protocols
- Auto-detects cohort and timestamp files
- Generates epoch-specific freezing percentages
- Outputs Excel files with multi-level column headers

**`ptc_staggered_ff.py`**
- Extended PTC processor for staggered timelines
- Inherits from base PTC class
- Handles delayed experiment initiation
- Applies temporal offset corrections per animal
- Automatically detects start_timestamps files

**`fr_ptc_data.py`**
- Specialized PTC data extraction
- Optimized for heatmap generation and custom visualization
- Processes multi-animal, multi-session datasets
- Extracts epoch-specific behavioral patterns

### Visualization Scripts

**`fr_pav_heatmaps.py`**
- Universal heatmap generation system
- Supports PTC, DTC, PTER, and staggered experiments
- Factory pattern architecture for experiment-specific processing
- Generates PNG heatmaps organized by group and animal
- Features:
  - Individual tone analysis with temporal windows
  - Color-coded freezing intensity visualization
  - Separate Training and LTM phase heatmaps
  - Automatic staggered timestamp detection

**`heatmaps_utils.py`**
- Visualization utility framework
- Provides base classes for data extraction and plotting
- Components:
  - `ExperimentDataExtractor`: Reads and organizes metadata
  - `ExperimentVisualizer`: Creates visualizations
  - `FreezeFrameDataProcessor`: Processes freeze-frame data
  - Specialized extractors for PTC, DTC, PTER

### Base Classes and Infrastructure

**`freezeframe.py`**
- Abstract FreezeFrame processing foundation
- Provides core functionality for all FreezeFrame processors
- Features:
  - Intelligent file detection
  - Cohort integration
  - Standardized column generation
  - Error handling and logging

**`utils.py`**
- Common utility functions
- `setup_logging()`: Configures file and console logging
- `plot_pair()`: Helper for highlighted plot regions
- Pipeline signature and metadata constants

### Application Scripts

**`app.py`**
- Graphical user interface entry point
- Launches experiment manager
- Features:
  - Visual experiment selection
  - Parameter configuration
  - Real-time progress monitoring
  - Integrated logging

**`driver.py`**
- Batch processing driver
- Configuration-driven execution via JSON
- Features:
  - Automated multi-experiment processing
  - Error recovery and continuation
  - Comprehensive audit trail
  - Dry-run mode for command preview

**`script_helper.py`**
- Interactive script assistance system
- Provides usage information and examples
- Shows required file structures
- Lists all available scripts with descriptions

---

## Reproduction Instructions

To reproduce the analyses from the associated manuscript:

### Step 1: Download Complete Dataset

Download the full experimental dataset from: [Data Repository URL - to be added]

### Step 2: Process PTC Experiments

```bash
python ptc_ff.py --folder "manuscript_data/PTC_Experiments" --output "manuscript_results/PTC_processed"
```

### Step 3: Generate Heatmaps

```bash
python fr_pav_heatmaps.py \
    --folder "manuscript_data/PTC_Experiments" \
    --output "manuscript_results/heatmaps" \
    --timestamps "manuscript_data/PTC_Timestamps.xlsx" \
    --experiment-type PTC
```

### Step 4: Statistical Analysis

Import processed Excel files into statistical software:
- **GraphPad Prism**: For figures and statistical tests described in Methods
- **Python/R**: For custom analyses using pandas/tidyverse
- **MATLAB**: For advanced temporal analysis

### Expected Results

- **Figure**: Heatmaps showing temporal freezing patterns
- **Figure**: Quantification of freezing responses across epochs
- **Supplementary Data**: Complete processed datasets

---

## Software License

This software is distributed under the **MIT License** (approved by the Open Source Initiative).

```
MIT License

Copyright (c) 2024 Shrestha Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Code Repository

The source code is available at:
- **GitHub**: https://github.com/preshre/ptc-behavior-pipeline.git

---

## Pseudocode Description

Detailed pseudocode describing the core algorithms is provided in:
- **Main Manuscript**: Methods section, "Behavioral Data Analysis"
- **Supplementary Methods**: Complete algorithmic descriptions
- **Inline Documentation**: Comprehensive docstrings in all Python files

---


## Contact & Support

**Developer**: Harshil Sanghvi  
**Laboratory**: Shrestha Lab, Stony Brook University  
**Department**: Neurobiology & Behavior  
**Principal Investigator**: Prof. Prerana Shrestha

For questions or support:
- Email: prerana.shrestha@stonybrook.edu

---

## Acknowledgments

This pipeline was developed in the Shrestha Lab at Stony Brook University for the analysis of Pavlovian threat conditioning behavioral data. We thank all lab members who contributed to testing and refinement of the analysis workflow.

**Development**: Harshil Sanghvi  
**Principal Investigator**: Prof. Prerana Shrestha  

---

## Version History

**v1.0.0** (2024) - Initial release
- Core PTC FreezeFrame processing
- Temporal heatmap generation
- Staggered timeline support
- Batch processing system