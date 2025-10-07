import os
import re
import csv
import time
import shutil
import subprocess
from pathlib import Path

# Paths
VIDEO_TO_360_PATH = "src/VideoTo360.py"
INPUT_DIRS = [
  "inputs4_experiments/synthetic1",
  # "inputs4_experiments/synthetic2",
  # "inputs4_experiments/synthetic3",
  # "inputs4_experiments/synthetic4",
]
RESULTS_DIR = Path("results1")
RESULTS_DIR.mkdir(exist_ok=True)

# Experiment configurations
EXPERIMENTS = [
  {"window_size": 40, "window_strategy": "simple", "relocalize": False},
  # {"window_size": 2, "window_strategy": "simple", "relocalize": False},
  # {"window_size": 5, "window_strategy": "simple", "relocalize": False},
  # {"window_size": 10, "window_strategy": "simple", "relocalize": False},
]

# Regex patterns for metrics
OUTPUT_DIR_PATTERN = re.compile(r"Output directory:\s*(.*)")
DIFF_PATTERN = re.compile(r"Average estimated vs inertial difference:\s*([\d.]+)\s*degrees")
COVERAGE_PATTERN = re.compile(r"Estimated orientation coverage:\s*([\d.]+)%")

# CSV file for results
CSV_PATH = RESULTS_DIR / "experiment_results.csv"
CSV_HEADERS = [
  "experiment_name",
  "input_dir",
  "window_size",
  "window_strategy",
  "relocalize",
  "avg_diff_degrees",
  "orientation_coverage_percent",
  "runtime_seconds",
]


def write_result_to_csv(row):
  """Append a single experiment result to the CSV file."""
  file_exists = CSV_PATH.exists()
  with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
    if not file_exists:
      writer.writeheader()
    writer.writerow(row)


def run_experiment(input_dir, window_size, window_strategy, relocalize):
  """Run a single experiment and collect results."""
  experiment_name = f"{Path(input_dir).name}_w{window_size}_{window_strategy}"
  if relocalize:
    experiment_name += "_relocalize"
  experiment_result_dir = RESULTS_DIR / experiment_name
  experiment_result_dir.mkdir(exist_ok=True)

  cmd = [
    "python",
    VIDEO_TO_360_PATH,
    input_dir,
    "--produce_debug",
    "--window_size",
    str(window_size),
    "--window_strategy",
    window_strategy,
  ]
  if relocalize:
    cmd.append("--relocalize")

  print(f"\n=== Running experiment: {experiment_name} ===")
  print("Command:", " ".join(cmd))

  start_time = time.time()
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  stdout, _ = process.communicate()
  runtime = time.time() - start_time

  # Save raw stdout log
  log_path = experiment_result_dir / "log.txt"
  log_path.write_text(stdout)

  # Extract metrics
  output_dir = None
  avg_diff = None
  coverage = None

  for line in stdout.splitlines():
    if m := OUTPUT_DIR_PATTERN.search(line):
      output_dir = m.group(1).strip()
    elif m := DIFF_PATTERN.search(line):
      avg_diff = float(m.group(1))
    elif m := COVERAGE_PATTERN.search(line):
      coverage = float(m.group(1))

  # Move output directory if found
  if output_dir and os.path.exists(output_dir):
    for item in os.listdir(output_dir):
      src = os.path.join(output_dir, item)
      dst = experiment_result_dir / item
      if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
      else:
        shutil.copy2(src, dst)
  else:
    print(f"⚠️  Warning: No output directory found for {experiment_name}")

  print(f"✅ Finished {experiment_name} | Avg diff: {avg_diff} | Coverage: {coverage} | Time: {runtime:.2f}s")

  result_row = {
    "experiment_name": experiment_name,
    "input_dir": input_dir,
    "window_size": window_size,
    "window_strategy": window_strategy,
    "relocalize": relocalize,
    "avg_diff_degrees": avg_diff,
    "orientation_coverage_percent": coverage,
    "runtime_seconds": round(runtime, 2),
  }

  # Write result immediately
  write_result_to_csv(result_row)

  return result_row

def main():
  for input_dir in INPUT_DIRS:
    for exp in EXPERIMENTS:
      run_experiment(input_dir, exp["window_size"], exp["window_strategy"], exp["relocalize"])

  print(f"\n✅ All experiments complete! Results saved incrementally to {CSV_PATH}")

if __name__ == "__main__":
  main()
