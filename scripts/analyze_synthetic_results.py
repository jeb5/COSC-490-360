import os
import re
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Paths
# RESULTS_DIR = Path("results1")
RESULTS_DIR = Path("results1")
INPUTS_DIR = Path("inputs4_experiments")
PLOT_DIR = RESULTS_DIR / "analysis_plots"
PLOT_DIR.mkdir(exist_ok=True)

# Regex to detect which synthetic dataset the experiment corresponds to
SYNTHETIC_REGEX = re.compile(r"synthetic(\d+)")


def load_orientations(path):
  """Load an orientations.csv file into a DataFrame indexed by frame number."""
  if not os.path.exists(path):
    print(f"⚠️ Missing file: {path}")
    return None
  df = pd.read_csv(path)
  df = df.rename(columns={"frame_number": "frame"})
  df = df[["frame", "pitch", "roll", "yaw"]]
  return df


def load_inertials(path):
  """Load inertials.csv (ground truth)."""
  df = pd.read_csv(path)
  df = df.rename(columns={"frame": "frame"})
  df = df[["frame", "pitch", "roll", "yaw"]]
  return df


def angular_difference(est_row, gt_row):
  """Compute angular difference in degrees between estimated and ground truth rotations."""
  # Convert from degrees to rotations
  r_est = R.from_euler("xyz", [est_row["pitch"], est_row["roll"], est_row["yaw"]], degrees=True)
  r_gt = R.from_euler("xyz", [gt_row["pitch"], gt_row["roll"], gt_row["yaw"]], degrees=True)
  # Relative rotation difference
  r_diff = r_est.inv() * r_gt
  # Angle magnitude in degrees
  return np.degrees(np.abs(r_diff.magnitude()))


def analyze_experiment(exp_path, inertials):
  """Compute inaccuracy over time for one experiment."""
  orientations_path = exp_path / "orientations.csv"
  est = load_orientations(orientations_path)
  if est is None:
    return None

  # Merge on frame number (inner join, to only keep frames with estimates)
  merged = pd.merge(est, inertials, on="frame", suffixes=("_est", "_gt"))
  inaccuracies = []
  for _, row in merged.iterrows():
    est_row = {"pitch": row["pitch_est"], "roll": row["roll_est"], "yaw": row["yaw_est"]}
    gt_row = {"pitch": row["pitch_gt"], "roll": row["roll_gt"], "yaw": row["yaw_gt"]}
    diff = angular_difference(est_row, gt_row)
    inaccuracies.append((row["frame"], diff))

  df_inacc = pd.DataFrame(inaccuracies, columns=["frame", "inaccuracy_deg"])
  # Ensure NaN gaps for missing frames (so matplotlib leaves breaks in the line)
  full_range = pd.Series(np.nan, index=np.arange(inertials["frame"].min(), inertials["frame"].max() + 1))
  full_range.loc[df_inacc["frame"]] = df_inacc["inaccuracy_deg"].values
  df_inacc_full = pd.DataFrame({"frame": full_range.index, "inaccuracy_deg": full_range.values})

  return df_inacc_full


experiment_names_lookup = {
  "w1_simple": "Rotation Chaining",
  "w40_simple": "Sliding Window w=40",
  "w40_overlapping": "Overlapping Windows w=40",
  "w40_overlapping_relocalize": "Overlapping Windows + Relocalization w=40",
  "w01_simple": "Rotation Chaining",
  "w02_simple": "Sliding Window w=2",
  "w05_simple": "Sliding Window w=5",
  "w10_simple": "Sliding Window w=10",
}


def main():
  # Find all experiments
  experiment_dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir() and (p / "orientations.csv").exists()]
  if not experiment_dirs:
    print("No experiments found with orientations.csv in results/")
    return

  # Group experiments by which synthetic input they correspond to
  experiments_by_synthetic = {}
  for exp_path in experiment_dirs:
    match = SYNTHETIC_REGEX.search(exp_path.name)
    if match:
      syn_num = match.group(1)
      experiments_by_synthetic.setdefault(syn_num, []).append(exp_path)
    else:
      print(f"⚠️ Could not determine synthetic dataset for {exp_path}")

  # Analyze each synthetic dataset
  for syn_num, exps in experiments_by_synthetic.items():
    inertials_path = INPUTS_DIR / f"synthetic{syn_num}" / "inertials.csv"
    inertials = load_inertials(inertials_path)
    if inertials is None:
      print(f"⚠️ Missing inertials.csv for synthetic{syn_num}")
      continue

    plt.figure(figsize=(10, 5))
    plt.title(f"Inaccuracy over Time — synthetic{syn_num}")
    plt.xlabel("Frame number")
    plt.ylabel("Inaccuracy (degrees)")

    for exp_path in sorted(exps):
      exp_name = exp_path.name
      exp_name = re.sub(r"^synthetic\d+_", "", exp_name)
      exp_name = experiment_names_lookup.get(exp_name, exp_name)
      df_inacc = analyze_experiment(exp_path, inertials)
      if df_inacc is None or df_inacc.empty:
        continue
      plt.plot(df_inacc["frame"], df_inacc["inaccuracy_deg"], label=exp_name)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_path = PLOT_DIR / f"synthetic{syn_num}_inaccuracy.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ Saved plot for synthetic{syn_num}: {save_path}")


if __name__ == "__main__":
  main()
