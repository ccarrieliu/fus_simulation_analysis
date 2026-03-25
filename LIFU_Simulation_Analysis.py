#####################################################################################
#   LIFU_Simulation_Analysis.py                                                     #
#   Carrie Liu                                                                      #
#   Last Updated March 2026                                                         #
##################################################################################### 

'''
Simulation_Analysis.py
-----------------------

This script contains the automated analysis and visualization pipeline for acoustic 
simulation data stored in HDF5 format. The user may request to 
process new simulation folders to extract 3D pressure fields and calculate 
critical performance metrics such as peak pressure, focal displacement, and 
FWHM focal volumes. This will trigger the generation of diagnostic plots, including 
beam profiles along principal axes and coronal/sagittal anatomy overlays, as well as 
the aggregation of all metrics into a standardized CSV file. All analysis routines 
and pltoting functions are included. 
'''

#------------import necessary packages for this script------------------------------#
import os                                                                           # System and file operations
from pathlib import Path                                                            # Object-oriented filesystem paths
import h5py                                                                         # HDF5 data format support
import matplotlib                                                                   # Plotting library
matplotlib.use("Agg")                                                               # Non-interactive backend for saving files
import matplotlib.pyplot as plt                                                     # Plotting interface
import numpy as np                                                                  # Numerical computing
import pandas as pd                                                                 # Data analysis and CSV handling
import yaml                                                                         # Configuration file parsing
from scipy import ndimage                                                           # Image processing and labeling
#####################################################################################

# ---------------------------- Plot styling ----------------------------------------#
plt.rcParams.update({                                                               # Set global plotting aesthetics
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 14,                    # Font sizes for labels and titles
    "xtick.labelsize": 11, "ytick.labelsize": 11,                                   # Font sizes for tick marks
})                                                                                  # 

# ------------------------------- Paths --------------------------------------------#
PRESSURE_ROOT = Path.home() / "Documents" / "simulation_analysis" / "pressure_results"


def ellipsoid_axes_from_mask(mask, spacing=(1.0, 1.0, 1.0), threshold=0.5):         
    """                                                                             
    Estimate principal ellipsoid diameters from a 3D binary mask.
    """         
    pts = np.argwhere(mask > threshold).astype(float)                               # Voxel coords of focal region
    if pts.shape[0] < 10: raise ValueError("Not enough voxels.")                    # Safety check for small regions
    pts_mm = pts * np.asarray(spacing, dtype=float)[None, :]                        # Convert voxels -> mm
    centroid = pts_mm.mean(axis=0)                                                  # Calculate geometric center
    centered = pts_mm - centroid[None, :]                                           # Zero-center the points
    covariance = (centered.T @ centered) / centered.shape[0]                        # Generate 2nd-moment matrix
    evals, evecs = np.linalg.eigh(covariance)                                       # Eigen-decomposition for axes
    order = np.argsort(evals)[::-1]                                                 # Sort major -> minor
    evals, evecs = evals[order], evecs[:, order]                                    # Reorder eigenvalues/vectors
    semi_axes = np.sqrt(5.0 * evals)                                                # Apply ellipsoid moment relation
    return {                                                                        # Return geometry dictionary
        "centroid_mm": centroid,                                                    # Center in mm
        "diameters_mm": 2.0 * semi_axes,                                            # Full diameters
        "semi_axes_mm": semi_axes,                                                  # Half-lengths
        "directions": evecs,                                                        # Orientation vectors
    }


def mask_cube(arr, center, cube_size=100):                                      
    """
    Keep only values inside a cube around the centre; assigns zero to all 
    elsewhere.
    """      
    half = cube_size // 2                                                           # Calculate radius
    z, y, x = center                                                                # Unpack center coordinates
    zmin, zmax = max(z - half, 0), min(z + half, arr.shape[0])                      # Bound Z axis
    ymin, ymax = max(y - half, 0), min(y + half, arr.shape[1])                      # Bound Y axis
    xmin, xmax = max(x - half, 0), min(x + half, arr.shape[2])                      # Bound X axis
    masked = np.zeros_like(arr)                                                     # Create empty template
    masked[zmin:zmax, ymin:ymax, xmin:xmax] = arr[zmin:zmax, ymin:ymax, xmin:xmax]  # Copy cube data
    return masked                                                                   # Return masked array


def get_amp_and_press_from_kdata(k_data_path):                                 
    """
    Extract the amplitude and pressure from k_data.yml, regardless of exact 
    nesting.
    """    
    with open(k_data_path, "r") as f: kd = yaml.safe_load(f)                        # Load configuration file
    amp = press = None                                                              # Initialize placeholders
    def walk(obj):                                                                  # Recursive walker function
        nonlocal amp, press                                                         # Access outer scope variables
        if isinstance(obj, dict):                                                   # 
            for k, v in obj.items():                                                # Iterate keys and values
                key = str(k).lower()                                                # Normalize key to lowercase
                if amp is None and key in ("amp", "amplitude"):                     # Check for amplitude key
                    try: amp = float(np.asarray(v).ravel()[0])                      # Cast value to float
                    except Exception: pass                                          # 
                if press is None and key in ("press", "pressure"):                  # Check for pressure key
                    try: press = float(np.asarray(v).ravel()[0])                    # Cast value to float
                    except Exception: pass                                          # 
                walk(v)                                                             # Recurse deeper
        elif isinstance(obj, list):                                                 # If object is a list
            for item in obj: walk(item)                                             # Recurse through items
    walk(kd)                                                                        # Start walking the YAML
    return np.nan if amp is None else amp, np.nan if press is None else press       # 


def calculate_pressure_offset(p_max_idx, focus_vox, dx):                        
    """
    Return distance in mm between max-pressure location and focus.
    """        
    p_pos = np.asarray(p_max_idx) * dx                                              # Max pressure location mm
    f_pos = np.asarray(focus_vox) * dx                                              # Intended focus location mm
    return np.linalg.norm(p_pos - f_pos)                                            # Euclidean distance


def loadNP(savepath, dtype=None):                                                 
    """
    Load a dict of numpy arrays from an .h5 file, resolving links.
    """        
    savepath = str(savepath)                                                        # Ensure path is string
    def try_cast(arr):                                                              # Helper for data typing
        if dtype is None or not isinstance(arr, np.ndarray): return arr             # Return as-is if no type
        return arr.astype(dtype, copy=False) if arr.dtype.kind in ("f", "i", "u") \
            else arr
    ret = {}                                                                        # Storage for file contents
    with h5py.File(savepath, "r") as h5f:                                           # Open HDF5 file
        for key in h5f:                                                             # Iterate datasets
            data = h5f[key][:]                                                      # Read data into memory
            if data.shape == (0,) and "pressure.h5" not in savepath:                # Case: Linked thermal data
                base = str(Path(savepath).parent.parent.parent)                     # Navigate directory tree
                for item in key.split("  "): base = os.path.join(base, item)        # Reconstruct linked path
                return loadNP(base, dtype=dtype)                                    # Recurse into linked file
            if data.shape == (2,) and "pressure.h5" in savepath:                    # Case: Linked acoustic data
                ky, scl = data                                                      # Key ID and scale factor
                for k in h5f:                                                       # Find matching link key
                    if str(int(ky)) + "  " in k: break                              # Match found
                while True:                                                         # Follow chain of links
                    base = os.getcwd()                                              # Start at current work dir
                    while "Arraynge" not in os.listdir(base):                       # Search upwards for project root
                        base = str(Path(base).parent)                               # Move up one level
                    base = os.path.join(base, "Arraynge", "Plan_And_Data", \
                                        "Data", "Results")
                    for item in k.split("  ")[1:]: base = os.path.join(base, item)  # Build link path
                    with h5py.File(base, "r") as linked:                            # Open the linked file
                        new_data = linked[str(int(ky))][:]                          # Get actual data array
                        if new_data.shape != (2,): break                            # Break if not another link
                        ky, scl = str(int(new_data[0])), scl * new_data[1]          # Update link and scale
                        for k in linked:                                            # Look for next link key
                            if str(int(ky)) + "  " in k: break                      # Match found
                ret[key] = try_cast(new_data * scl)                                 # Store scaled result
                continue                                                            # 
            if "  " not in key and data.dtype != "O": ret[key] = try_cast(data)     # Regular numeric storage
            elif "  " not in key and data.dtype == "O":                             # Case: Object/String data
                ret[key] = data                                                     # Store object array
                for i in range(len(ret[key])): ret[key][i] = str(ret[key][i])[2:-1] # Clean string formatting
    return ret                                                                      # Return full data dict


def get_first_3d_array(data_dict, name="unknown"):                         
    """
    Return the first 3D array found in a dictionary.
    """                     
    for key, value in data_dict.items():                                            # Loop through items
        if isinstance(value, np.ndarray) and value.ndim == 3:                       # Check for 3 dimensions
            print(f"Found 3D array for '{name}' under key '{key}'")                 # 
            return value                                                            # Return the 3D volume
    raise ValueError(f"No 3D array found in dict for '{name}'.")                    # 


def save_principal_axes_plot(results_folder, sub_id, target, focus_mask, dx_val, \
                             pressure, pressure_masked, input_folder):
    """
    Plot beam profiles along the fitted principal axes.
    """               
    ellipsoid = ellipsoid_axes_from_mask(focus_mask, spacing=(dx_val, dx_val, dx_val))
    directions, centroid_mm = ellipsoid["directions"], ellipsoid["centroid_mm"]     # Extract geometry
    t_steps = np.linspace(-40, 40, 200)                                             # Range: +/- 40mm
    fig, ax = plt.subplots(figsize=(8, 6))                                          # Initialize figure
    axis_names = ["Axial (Long)", "Lateral (Mid)", "Lateral (Minor)"]               # Label strings
    colors = ["#FF4136", "#0074D9", "#2ECC40"]                                # Colors
    for i, name in enumerate(axis_names):                                           # Iterate axes
        line_mm = centroid_mm[None, :] + t_steps[:, None] * directions[:, i]        # Line points in mm
        line_vox = (line_mm / dx_val).astype(int)                                   # Points in voxels
        for dim in range(3): line_vox[:, dim] = np.clip(line_vox[:, dim], 0, \
                                                        pressure.shape[dim] - 1)
        profile = pressure_masked[line_vox[:, 0], line_vox[:, 1], line_vox[:, 2]]   # Sample pressure
        ax.plot(t_steps, profile, label=name, color=colors[i], linewidth=2)         # Add to plot
    ax.set(title=f"Beam Profiles Along Principal Axes", xlabel="Dist [mm]", \
           ylabel="Press [MPa]")
    ax.legend()                                                                 
    plt.savefig(Path(results_folder) / f"S{sub_id}T{target}_principal_axes.png", \
                dpi=300)
    plt.close(fig)                                                              


def save_anatomy_overlay(results_folder, sub_id, target, pressure, lyn, dx, p_max_idx):
    """
    Save coronal and sagittal pressure overlays on anatomy.
    """
    x0, y0, z0 = map(int, p_max_idx)                                               # Focal point coordinates
    p_cor, a_cor = pressure[:, y0, :] * 1e-6, lyn[:, y0, :]                        # Coronal slices
    p_sag, a_sag = pressure[x0, :, :] * 1e-6, lyn[x0, :, :]                        # Sagittal slices
    pmax = max(p_cor.max(), p_sag.max())                                           # Max for color scaling
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)      # Two-pane layout
    ext_c = [0, pressure.shape[0] * dx, 0, pressure.shape[2] * dx]                 # Coronal dimensions
    ext_s = [0, pressure.shape[1] * dx, 0, pressure.shape[2] * dx]                 # Sagittal dimensions
    axes[0].imshow(np.rot90(a_cor, 1), alpha=0.30, cmap="gray", extent=ext_c)      # Background anatomy
    im0 = axes[0].imshow(np.rot90(p_cor, 1), alpha=0.70, cmap="hot", \
                         extent=ext_c, vmin=0, vmax=pmax)
    axes[0].set_title("Coronal"); axes[0].set_xlabel("x [mm]"); \
        axes[0].set_ylabel("z [mm]")
    axes[1].imshow(np.rot90(a_sag, 1), alpha=0.30, cmap="gray", extent=ext_s)      # Background anatomy
    axes[1].imshow(np.rot90(p_sag, 1), alpha=0.70, cmap="hot", extent=ext_s, \
                   vmin=0, vmax=pmax)
    axes[1].set_title("Sagittal"); axes[1].set_xlabel("y [mm]")                    # Axis titles
    fig.colorbar(im0, ax=axes, shrink=0.95, label="Pressure [MPa]")                # Shared colorbar
    plt.savefig(Path(results_folder) / f"S{sub_id}T{target}_anatomy_overlay.png", \
                dpi=300, bbox_inches="tight")
    plt.close(fig)                                                                 


def save_pressure_profiles_plot(results_folder, sub_id, target, pressure, dx, x0, y0, z0):
    """
    Plot x/y/z pressure profiles through the max-pressure voxel.
    """        
    z_profile = pressure[x0, y0, :] * 1e-6                                         # Axial and lateral X, Y profiles
    y_profile = pressure[x0, :, z0] * 1e-6                                         # 
    x_profile = pressure[:, y0, z0] * 1e-6                                         # 
    z_mm = (np.arange(len(z_profile)) - z0) * dx                                   # Centers of X, Y, Z at max
    y_mm = (np.arange(len(y_profile)) - y0) * dx                                   # 
    x_mm = (np.arange(len(x_profile)) - x0) * dx                                   # 
    fig, ax = plt.subplots(figsize=(8, 6))                                         # 
    ax.plot(x_mm, x_profile, label="Lateral (X)", linewidth=1.5, color="#0074D9")# Plot X
    ax.plot(y_mm, y_profile, label="Lateral (Y)", linewidth=1.5, color="#FF4136")# Plot Y
    ax.plot(z_mm, z_profile, label="Axial (Z)", linewidth=1.5, color="#2ECC40")  # Plot Z
    ax.set(title="Pressure Profiles", xlabel="Dist from Max [mm]", \
           ylabel="Pressure [MPa]")
    ax.grid(axis="both", alpha=0.3, linestyle="--")
    ax.legend(); ax.set_xlim(-60, 60)
    plt.tight_layout()                                                             # Minimize whitespace
    plt.savefig(Path(results_folder) / \
        f"S{sub_id}T{target}_pressure_profiles.png", dpi=300)
    plt.close(fig)                                                                 


def find_sims_to_process(root):                                                    
    """
    Return Sim folders newer than the last analyzed one.
    """                 
    sim_folders = \
        [f for f in os.listdir(root) if f.startswith("Sim") and f[3:].isdigit()]
    if not sim_folders: return []                                                  # Early exit if empty
    sim_nums = sorted((int(f[3:]) for f in sim_folders), reverse=True)             # Sort descending
    last_analyzed = 0                                                              # Start search from zero
    for sim_num in sim_nums:                                                       # Check for analysis.csv
        if (root / f"Sim{sim_num}" / "results" / "analysis.csv").exists():         # Found analyzed folder
            last_analyzed = sim_num                                                # Update index
            print(f"Found existing analysis up to: Sim{last_analyzed}")            # 
            break                                                                  # 
    return sorted(n for n in sim_nums if n > last_analyzed)                        # Return new sims


def build_focus_mask(pressure_masked):                                          
    """
    Create a focal-region mask using descending relative thresholds.
    """     
    p_max = np.nanmax(pressure_masked)                                             # Find peak value
    struct = ndimage.generate_binary_structure(3, 2)                               # Define 3D connectivity
    for threshold in [0.50, 0.40, 0.30, 0.20, 0.10]:                               # Try thresholds high to low
        mask = pressure_masked >= threshold * p_max                                # Apply threshold
        labeled, num_features = ndimage.label(mask, structure=struct)              # Group connected pixels
        if num_features == 0: continue                                             # 
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))             # Measure component sizes
        candidate = labeled == int(np.argmax(sizes) + 1)                           # Select largest component
        if np.count_nonzero(candidate) >= 10: return candidate                     # Return if large enough
    return pressure_masked >= 0.10 * p_max                                         # Absolute fallback


if __name__ == "__main__":                                                      
    """
    Generate pressure plots and summary metrics for each new simulation.
    """  
    to_process = find_sims_to_process(PRESSURE_ROOT)                               # Identify new simulations
    if not to_process:                                                             # 
        print("All simulations are already analyzed. Nothing to do!")              # 
        raise SystemExit                                                           # 
    print(f"Simulations queued for analysis: {to_process}")                        # 
    sub_ids, targets = ["07"], ["1"]                                               # Simulation parameters
    for input_sim in to_process:                                                   # Loop through simulations
        input_folder = f"Sim{input_sim}"                                           # Set folder name
        print(f"\n>>> Processing {input_folder}...")                               # 
        results_folder = PRESSURE_ROOT / input_folder / "results"                  # Define output path
        results_folder.mkdir(parents=True, exist_ok=True)                          # Ensure folder exists
        output_file = results_folder / "analysis.csv"                              # Target CSV path
        rows = []                                                                  # Row storage for CSV
        for sub_id in sub_ids:                                                     # Nested loop: Subjects
            for target in targets:                                                 # Nested loop: Targets
                info = {"sub_id": sub_id, "target": target}                        # Start row metadata
                sim_folder = PRESSURE_ROOT / input_folder / f"S{sub_id}T{target}"  # Local simulation path
                pressure_path, k_data_path = \
                    sim_folder / "pressure.h5", sim_folder / "k_data.yml"
                anatomy_path, thermal_path = \
                    sim_folder / "anatomy.h5", sim_folder / "thermal.h5"
                pressure = \
                    get_first_3d_array(loadNP(pressure_path), name="pressure_path")
                anatomy_dict = loadNP(anatomy_path)                                # Load anatomy data
                lyn = get_first_3d_array(anatomy_dict, name="anatomy_path")        # Get anatomy volume
                dx = anatomy_dict.get("dx", None)                                  # Get resolution
                if dx is None: raise ValueError(f"No 'dx' in {anatomy_path}")      # Validate dx
                dx_val = float(np.asarray(dx).ravel()[0])                          # Convert to float
                with open(k_data_path, "r") as f: k_data = yaml.safe_load(f)       # Load target config
                focus_mm = np.asarray(k_data["Focus 1"]["f_pos"], dtype=float)     # Get intended focus mm
                focus_vox = np.rint(focus_mm / dx_val).astype(int)                 # Focus in voxels
                if thermal_path.exists(): thermal_path.unlink()                    # Clean up old temporary files
                pressure_masked = pressure * 1e-6                                  # Convert Pa to MPa
                p_max_idx = \
                    np.unravel_index(np.argmax(pressure_masked), \
                        pressure_masked.shape)
                p_max = np.nanmax(pressure_masked)                                 # Peak pressure
                focus_mask = build_focus_mask(pressure_masked)                     # Get focal volume mask
                centroid = np.array(ndimage.center_of_mass(focus_mask)).astype(int)# Find center of mass
                p_centroid = np.round(pressure[tuple(centroid)], 3)                # Pressure at center
                foc_dist = np.round(np.linalg.norm(focus_vox - centroid) * dx_val, 3) # Center-to-target distance
                fwhm_vol = np.round(np.count_nonzero(focus_mask) * dx_val**3, 1)   # FWHM volume in mm^3
                axis_lengths = ellipsoid_axes_from_mask(focus_mask, \
                    spacing=(dx_val, dx_val, dx_val))["diameters_mm"]
                ax1, ax2, ax3 = np.round(axis_lengths, 2)                          # Unpack major/minor axes
                amp, press_val = get_amp_and_press_from_kdata(k_data_path)         # Get input/output stats
                info.update({                                                      # Populate results dict
                    "maxP": p_max, "centP": p_centroid, "cent_foc_dist": foc_dist,
                    "focus_displacement_mm": \
                        calculate_pressure_offset(p_max_idx, focus_vox, dx_val),
                    "fwhm_vol": fwhm_vol, 
                    "fwhm_axis1": ax1, 
                    "fwhm_axis2": ax2, 
                    "fwhm_axis3": ax3,
                    "amp_kdata": amp, "press_kdata": press_val,
                    "gain_press_over_amp": press_val / amp if np.isfinite(amp) \
                        and amp != 0 else np.nan,
                })                                                                 # End of dict update
                rows.append(info)                                                  # Append to batch list
                save_principal_axes_plot(results_folder, sub_id, target, \
                    focus_mask, dx_val, pressure, pressure_masked, input_folder)
                save_anatomy_overlay(results_folder, sub_id, target, pressure, \
                    lyn, dx_val, p_max_idx)
                save_pressure_profiles_plot(results_folder, sub_id, target, \
                    pressure, dx_val, *p_max_idx)
                print(f"Finished processing S{sub_id}T{target}")                   # Individual status update
        pd.DataFrame(rows).to_csv(output_file, index=False)                        # Export results to CSV
        print(f"Analysis saved to {output_file}")                                  # File save log
    print("\nBatch analysis complete.")                                            # Final completion log