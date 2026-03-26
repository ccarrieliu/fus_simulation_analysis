
#########################################################################################
#   Cross-simulation_Comparative_Analysis.py                                            #
#   Carrie Liu                                                                          #
#   Last Updated March 2026                                                             #
######################################################################################### 

'''
Cross-simulation_Comparative_Analysis.py
------------------------------------------

This script provides the automated analysis and visualization pipeline for acoustic 
simulation data stored in HDF5 format. It extracts 3D pressure fields and anatomical 
data to calculate metrics like peak pressure, focal displacement, and FWHM volumes. 
The routine generates diagnostic plots—including beam profiles and anatomy overlays—
and aggregates all results into a standardized CSV file for batch processing.
'''

#------------import necessary packages for this script----------------------------------#
import os                                                                               # System and file operations
from pathlib import Path                                                                # Object-oriented filesystem paths
import h5py                                                                             # HDF5 data format support
import matplotlib                                                                       # Plotting library
matplotlib.use("Agg")                                                                   # Non-interactive backend
import matplotlib.pyplot as plt                                                         # Plotting interface
import numpy as np                                                                      # Numerical computing
import pandas as pd                                                                     # Data analysis and CSV handling
import yaml                                                                             # Configuration file parsing
from scipy import ndimage                                                               # Image processing and labeling

# ---------------------------- Plot styling --------------------------------------------#
plt.rcParams.update({                                                                   # Set global plotting aesthetics
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 14,                        # Font sizes for labels and titles
    "xtick.labelsize": 11, "ytick.labelsize": 11,                                       # Font sizes for tick marks
})                                                                                      # End of rcParams update

# ------------------------------- Paths ------------------------------------------------#
PRESSURE_ROOT = Path.home() / "Documents" / "simulation_analysis" / "pressure_results" 

def ellipsoid_axes_from_mask(mask, spacing=(1.0, 1.0, 1.0), threshold=0.5):             # Estimates 3D ellipsoid diameters
    pts = np.argwhere(mask > threshold).astype(float)                                   # Voxel coords of focal region
    if pts.shape[0] < 10: raise ValueError("Not enough voxels.")                        # Safety check
    pts_mm = pts * np.asarray(spacing, dtype=float)[None, :]                            # Convert voxels -> mm
    centroid = pts_mm.mean(axis=0)                                                      # Calculate geometric center
    centered = pts_mm - centroid[None, :]                                               # Zero-center the points
    covariance = (centered.T @ centered) / centered.shape[0]                            # Generate 2nd-moment matrix
    evals, evecs = np.linalg.eigh(covariance)                                           # Eigen-decomposition for axes
    order = np.argsort(evals)[::-1]                                                     # Sort major -> minor
    evals, evecs = evals[order], evecs[:, order]                                        # Reorder eigenvalues/vectors
    semi_axes = np.sqrt(5.0 * evals)                                                    # Apply ellipsoid moment relation
    return {                                                                            # Return geometry dictionary
        "centroid_mm": centroid, "diameters_mm": 2.0 * semi_axes,                       # Center and full diameters
        "semi_axes_mm": semi_axes, "directions": evecs,                                 # Half-lengths and orientation
    }

def mask_cube(arr, center, cube_size=100):                                              # Isolates a sub-volume cube
    half = cube_size // 2                                                               # Calculate radius
    z, y, x = center                                                                    # Unpack center coordinates
    zmin, zmax = max(z - half, 0), min(z + half, arr.shape[0])                          # Bound Z axis
    ymin, ymax = max(y - half, 0), min(y + half, arr.shape[1])                          # Bound Y axis
    xmin, xmax = max(x - half, 0), min(x + half, arr.shape[2])                          # Bound X axis
    masked = np.zeros_like(arr)                                                         # Create empty template
    masked[zmin:zmax, ymin:ymax, xmin:xmax] = arr[zmin:zmax, ymin:ymax, xmin:xmax]      # Copy cube data
    return masked                                                                       # Return masked array

def get_amp_and_press_from_kdata(k_data_path):                                          # Scans YAML for specific keys
    with open(k_data_path, "r") as f: kd = yaml.safe_load(f)                            # Load configuration file
    amp = press = None                                                                  # Initialize placeholders
    def walk(obj):                                                                      # Recursive walker function
        nonlocal amp, press                                                             # Access outer scope variables
        if isinstance(obj, dict):                                                       # If object is a dictionary
            for k, v in obj.items():                                                    # Iterate keys and values
                key = str(k).lower()                                                    # Normalize key to lowercase
                if amp is None and key in ("amp", "amplitude"):                         # Check for amplitude key
                    try: amp = float(np.asarray(v).ravel()[0])                          # Cast value to float
                    except Exception: pass                                              # Skip on error
                if press is None and key in ("press", "pressure"):                      # Check for pressure key
                    try: press = float(np.asarray(v).ravel()[0])                        # Cast value to float
                    except Exception: pass                                              # Skip on error
                walk(v)                                                                 # Recurse deeper
        elif isinstance(obj, list):                                                     # If object is a list
            for item in obj: walk(item)                                                 # Recurse through items
    walk(kd)                                                                            # Start walking the YAML
    return np.nan if amp is None else amp, np.nan if press is None else press           # Return found values or NaN

def calculate_pressure_offset(p_max_idx, focus_vox, dx):                                # Measures target accuracy
    return np.linalg.norm(np.asarray(p_max_idx) * dx - np.asarray(focus_vox) * dx)      # Euclidean distance in mm

def loadNP(savepath, dtype=None):                                                       # Robust HDF5 loader
    savepath = str(savepath)                                                            # Ensure path is string
    def try_cast(arr):                                                                  # Helper for data typing
        if dtype is None or not isinstance(arr, np.ndarray):
            return arr                                                                  # Return as-is if no type or not numpy array
        if arr.dtype.kind in ("f", "i", "u"):
            return arr.astype(dtype, copy=False)
        return arr
    ret = {}                                                                            # Storage for file contents
    with h5py.File(savepath, "r") as h5f:                                               # Open HDF5 file
        for key in h5f:                                                                 # Iterate datasets
            data = h5f[key][:]                                                          # Read data into memory
            if data.shape == (0,) and "pressure.h5" not in savepath:                    # Case: Linked thermal data
                base = str(Path(savepath).parent.parent.parent)                         # Navigate directory tree
                for item in key.split("  "): base = os.path.join(base, item)            # Reconstruct linked path
                return loadNP(base, dtype=dtype)                                        # Recurse into linked file
            if data.shape == (2,) and "pressure.h5" in savepath:                        # Case: Linked acoustic data
                ky, scl = data                                                          # Key ID and scale factor
                for k in h5f:                                                           # Find matching link key
                    if str(int(ky)) + "  " in k: break                                  # Match found
                while True:                                                             # Follow chain of links
                    base = os.getcwd()                                                  # Start at current work dir
                    while "Arraynge" not in os.listdir(base): base = str(Path(base).parent)  # Search for project root
                    base = os.path.join(base, "Arraynge", "Plan_And_Data", "Data", "Results")
                    for item in k.split("  ")[1:]: base = os.path.join(base, item)      # Build link path
                    with h5py.File(base, "r") as linked:                                # Open the linked file
                        new_data = linked[str(int(ky))][:]                              # Get actual data array
                        if new_data.shape != (2,): break                                # Break if not another link
                        ky, scl = str(int(new_data[0])), scl * new_data[1]              # Update link and scale
                        for k in linked:                                                # Look for next link key
                            if str(int(ky)) + "  " in k: break                          # Match found
                ret[key] = try_cast(new_data * scl); continue                           # Store scaled result
            if "  " not in key and data.dtype != "O": ret[key] = try_cast(data)         # Regular numeric storage
            elif "  " not in key and data.dtype == "O":                                 # Case: Object/String data
                ret[key] = data                                                         # Store object array
                for i in range(len(ret[key])): ret[key][i] = str(ret[key][i])[2:-1]     # Clean string formatting
    return ret                                                                          # Return full data dict

def get_first_3d_array(data_dict, name="unknown"):                                      # Utility to find volume data
    for key, value in data_dict.items():                                                # Loop through items
        if isinstance(value, np.ndarray) and value.ndim == 3:                           # Check for 3 dimensions
            print(f"Found 3D array for '{name}' under key '{key}'")                     # Log discovery
            return value                                                                # Return the 3D volume
    raise ValueError(f"No 3D array found in dict for '{name}'.")                        # Error if missing

def save_principal_axes_plot(results_folder, sub_id, target, focus_mask, dx_val, pressure, pressure_masked, input_folder):
    ellipsoid = ellipsoid_axes_from_mask(focus_mask, spacing=(dx_val, dx_val, dx_val))
    directions, centroid_mm = ellipsoid["directions"], ellipsoid["centroid_mm"]         # Extract geometry
    t_steps = np.linspace(-40, 40, 200)                                                 # Range: +/- 40mm
    fig, ax = plt.subplots(figsize=(8, 6))                                              # Initialize figure
    axis_names = ["Axial (Long)", "Lateral (Mid)", "Lateral (Minor)"]                   # Label strings
    colors = ["#FF4136", "#0074D9", "#2ECC40"]                                    # Distinct colors
    for i, name in enumerate(axis_names):                                               # Iterate axes
        line_mm = centroid_mm[None, :] + t_steps[:, None] * directions[:, i]            # Line points in mm
        line_vox = (line_mm / dx_val).astype(int)                                       # Points in voxels
        for dim in range(3): 
            line_vox[:, dim] = np.clip(line_vox[:, dim], 0, pressure.shape[dim] - 1)
        profile = pressure_masked[line_vox[:, 0], line_vox[:, 1], line_vox[:, 2]]       # Sample pressure
        ax.plot(t_steps, profile, label=name, color=colors[i], linewidth=2)             # Add to plot
    ax.set(title=f"Beam Profiles Along Principal Axes", 
        xlabel="Distance [mm]", ylabel="Pressure [MPa]")
    ax.legend()
    plt.savefig(Path(results_folder) / f"S{sub_id}T{target}_principal_axes.png", dpi=300)
    plt.close(fig)

def save_anatomy_overlay(results_folder, sub_id, target, pressure, lyn, dx, p_max_idx):
    x0, y0, z0 = map(int, p_max_idx)                                                    # Focal point coordinates
    p_cor, a_cor = pressure[:, y0, :] * 1e-6, lyn[:, y0, :]                             # Coronal slices
    p_sag, a_sag = pressure[x0, :, :] * 1e-6, lyn[x0, :, :]                             # Sagittal slices
    pmax = max(p_cor.max(), p_sag.max())                                                # Max for scaling
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)           # Two-pane layout
    ext_c = [0, pressure.shape[0] * dx, 0, pressure.shape[2] * dx]                      # Coronal dimensions
    ext_s = [0, pressure.shape[1] * dx, 0, pressure.shape[2] * dx]                      # Sagittal dimensions
    axes[0].imshow(np.rot90(a_cor, 1), alpha=0.30, cmap="gray", extent=ext_c)           # Background anatomy
    im0 = axes[0].imshow(np.rot90(p_cor, 1), alpha=0.70,
        cmap="hot", extent=ext_c, vmin=0.0, vmax=pmax)
    axes[0].set_title("Coronal"); axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("z [mm]")
    axes[1].imshow(np.rot90(a_sag, 1), alpha=0.30, cmap="gray", extent=ext_s)           # Background anatomy
    axes[1].imshow(np.rot90(p_sag, 1), alpha=0.70, cmap="hot", 
            extent=ext_s, vmin=0.0, vmax=pmax)
    axes[1].set_title("Sagittal"); axes[1].set_xlabel("y [mm]")                         # Axis titles
    fig.colorbar(im0, ax=axes, shrink=0.95, label="Pressure [MPa]")                     # Shared colorbar
    plt.savefig(Path(results_folder) / f"S{sub_id}T{target}_anatomy_overlay.png", 
        dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_pressure_profiles_plot(results_folder, sub_id, target, pressure, dx, x0, y0, z0):
    (z_profile, y_profile, x_profile) = (
        pressure[x0, y0, :] * 1e-6,
        pressure[x0, :, z0] * 1e-6, 
        pressure[:, y0, z0] * 1e-6
    )
    (z_mm, y_mm, x_mm) = (
        (np.arange(len(z_profile)) - z0) * dx,
        (np.arange(len(y_profile)) - y0) * dx,
        (np.arange(len(x_profile)) - x0) * dx
    )
    fig, ax = plt.subplots(figsize=(8, 6))                                              # Init plot
    ax.plot(x_mm, x_profile, label="Lateral (X)", color="#0074D9")                    # Plot X
    ax.plot(y_mm, y_profile, label="Lateral (Y)", color="#FF4136")                    # Plot Y
    ax.plot(z_mm, z_profile, label="Axial (Z)", color="#2ECC40")                      # Plot Z
    ax.set(title="Pressure Profiles", xlabel="Dist from Max [mm]", ylabel="Pressure [MPa]")
    ax.grid(axis="both", alpha=0.3, linestyle="--"); ax.legend()
    ax.set_xlim(-60, 60); plt.tight_layout()
    plt.savefig(Path(results_folder) / f"S{sub_id}T{target}_anchored_profiles.png", 
        dpi=300, bbox_inches="tight")
    plt.close(fig)

def find_sims_to_process(root):                                                         # Filters out existing analysis
    sim_folders = [f for f in os.listdir(root) if f.startswith("Sim") and f[3:].isdigit()]
    if not sim_folders: return []                                                       # Early exit
    sim_nums = sorted((int(f[3:]) for f in sim_folders), reverse=True)                  # Sort descending
    last_analyzed = 0                                                                   # Start search from zero
    for sim_num in sim_nums:                                                            # Check for analysis.csv
        if (root / f"Sim{sim_num}" / "results" / "analysis.csv").exists():              # Found analyzed folder
            last_analyzed = sim_num; break                                              # Update index and stop
    return sorted(n for n in sim_nums if n > last_analyzed)                             # Return new sims

def build_focus_mask(pressure_masked):                                                  # Creates binary focal volume
    p_max = np.nanmax(pressure_masked)
    structure = ndimage.generate_binary_structure(3, 2)
    for threshold in [0.50, 0.40, 0.30, 0.20, 0.10]:                                    # Try thresholds high to low
        mask = pressure_masked >= threshold * p_max                                     # Apply threshold
        labeled, num_features = ndimage.label(mask, structure=structure)                # Group connected pixels
        if num_features == 0: continue                                                  # Try next if empty
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))                  # Measure component sizes
        candidate = labeled == int(np.argmax(sizes) + 1)                                # Select largest component
        if np.count_nonzero(candidate) >= 10: return candidate                          # Return if large enough
    return pressure_masked >= 0.10 * p_max                                              # Absolute fallback

if __name__ == "__main__":                                                              # Main execution block
    to_process = find_sims_to_process(PRESSURE_ROOT)                                    # Identify new simulations
    if not to_process: print("Nothing to analyze."); raise SystemExit                   # Graceful exit
    sub_ids, targets = ["07"], ["1"]                                                    # Simulation parameters
    for input_sim in to_process:                                                        # Loop through simulations
        input_folder = f"Sim{input_sim}"; print(f">>> Processing {input_folder}")       # Header log
        results_folder = PRESSURE_ROOT / input_folder / "results"                       # Define output path
        results_folder.mkdir(parents=True, exist_ok=True); rows = []                    # Ensure folder exists
        for sub_id in sub_ids:                                                          # Nested loop: Subjects
            for target in targets:                                                      # Nested loop: Targets
                info = {"sub_id": sub_id, "target": target}                             # Start row metadata
                sim_folder = PRESSURE_ROOT / input_folder / f"S{sub_id}T{target}"       # Simulation path
                pressure_path = sim_folder / "pressure.h5"
                k_data_path = sim_folder / "k_data.yml"
                anatomy_path = sim_folder / "anatomy.h5"
                thermal_path = sim_folder / "thermal.h5"
                pressure = get_first_3d_array(loadNP(pressure_path), name="pressure_path")
                anatomy_dict = loadNP(anatomy_path)
                lyn = get_first_3d_array(anatomy_dict, name="anatomy_path")
                dx = anatomy_dict.get("dx", None); dx_val = float(np.asarray(dx).ravel()[0])
                with open(k_data_path, "r") as f: k_data = yaml.safe_load(f)            # Load target config
                focus_mm = np.asarray(k_data["Focus 1"]["f_pos"], dtype=float)          # Get focus mm
                focus_vox = np.rint(focus_mm / dx_val).astype(int)                      # Focus in voxels
                if thermal_path.exists(): thermal_path.unlink()                         # Clean up old files
                pressure_masked = pressure * 1e-6                                       # Convert Pa to MPa
                p_max_idx = np.unravel_index(np.argmax(pressure_masked), pressure_masked.shape)
                p_max, focus_mask = np.nanmax(pressure_masked), build_focus_mask(pressure_masked)
                centroid = np.array(ndimage.center_of_mass(focus_mask)).astype(int)     # Find center of mass
                foc_centroid_dist = np.round(np.linalg.norm(focus_vox - centroid) * dx_val, 3)   # Target distance
                fwhm_volume = np.round(np.count_nonzero(focus_mask) * dx_val**3, 1)     # FWHM volume
                axis_lengths = ellipsoid_axes_from_mask(focus_mask, 
                    spacing=(dx_val, dx_val, dx_val))["diameters_mm"]
                axis_1, axis_2, axis_3 = np.round(axis_lengths, 2)
                amp, press_val = get_amp_and_press_from_kdata(k_data_path)
                info.update({"maxP": p_max, 
                             "centP": np.round(pressure[tuple(centroid)], 3), 
                             "cent_foc_dist": foc_centroid_dist,
                            "focus_displacement_mm": calculate_pressure_offset(p_max_idx, focus_vox, dx_val),
                            "fwhm_vol": fwhm_volume, "fwhm_axis1": axis_1, 
                            "fwhm_axis2": axis_2, "fwhm_axis3": axis_3, "amp_kdata": amp, 
                            "press_kdata": press_val, 
                            "gain_press_over_amp": press_val / amp if np.isfinite(amp) and amp != 0 else np.nan})
                rows.append(info)
                save_principal_axes_plot(results_folder, sub_id, target, focus_mask, 
                                        dx_val, pressure, pressure_masked, input_folder)
                save_anatomy_overlay(results_folder, sub_id, target, pressure, lyn, 
                                     dx_val, p_max_idx)
                x0, y0, z0 = map(int, p_max_idx)
                save_pressure_profiles_plot(results_folder, sub_id, target, pressure, 
                                            dx_val, x0, y0, z0)
                print(f"Finished S{sub_id}T{target}")
        pd.DataFrame(rows).to_csv(results_folder / "analysis.csv", index=False)         # Export results to CSV
    print("\nBatch analysis complete.")                                                 # Final completion log