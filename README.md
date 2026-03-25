# Focused Ultrasound Simulation Analysis

## Overview
This project analyzes pressure fields from transcranial focused ultrasound (FUS) simulation results, performed using a private k-wave based platform. 
It extracts beam characteristics to perform beam profiling, focal volume estimation, and anatomical visualization, which are logged and outputted as figures. 

## Motivation
Accurate targeting in transcranial FUS is challenging due to skull-induced distortion.
This project builds tools to quantify beam focusing quality and analyze simulation outputs.

## Features
- Extracts 3D pressure fields from `.h5` files
- Computes:
  - Maximum pressure
  - Focal centroid
  - Focus displacement
  - FWHM volume
  - Principal axis beam profiles
- Generates:
  - Pressure profile plots
  - Anatomy overlays
  - Beam axis visualizations

## Tech Stack
- Python (NumPy, SciPy, h5py)
- Matplotlib
- YAML parsing

## Example Output
![Anatomy Figure](docs/S07T1_anatomy_overlay.png)
![Principal Axes Plot](docs/S07T1_Principal_axes.png)
![Pressure Profiles Plot](docs/S07T1_anchored_profiles.png)
![Analysis CSV File](docs/analysis.csv)

## How to Run
```bash
pip install -r requirements.txt
python scripts/run_analysis.py
