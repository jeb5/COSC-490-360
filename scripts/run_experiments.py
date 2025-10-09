import os
import re
import csv
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
VIDEO_TO_360_PATH = "src/VideoTo360.py"
INPUT_DIRS = [str(p) for p in Path("inputs3").iterdir() if p.is_dir()]
print(f"Running experiments on input directories: {INPUT_DIRS}")
# INPUT_DIRS = []
RESULTS_DIR = Path("results4")
RESULTS_DIR.mkdir(exist_ok=True)

# Parallel execution settings
PARALLEL_EXPERIMENTS = 3  # Number of experiments to run in parallel

# Experiment configurations
EXPERIMENTS = [
  {"use_inertials": True},
  {"window_size": 1, "window_strategy": "simple"},
  {"window_size": 40, "window_strategy": "simple"},
  {"window_size": 40, "window_strategy": "quadratic"},
  {"window_size": 40, "window_strategy": "overlapping", "relocalize": True},
  {"window_size": 40, "window_strategy": "overlapping", "relocalize": True},
  {"window_size": 150, "window_strategy": "overlapping", "relocalize": True},
]
default_args = {
  "produce_debug": True,
  "produce_360": True,
  "output_scale": 0.5,
}
# Add default arguments to all experiments
EXPERIMENTS = [{**default_args, **exp} for exp in EXPERIMENTS]


# Regex patterns for metrics
OUTPUT_DIR_PATTERN = re.compile(r"Output directory:\s*(.*)")
DIFF_PATTERN = re.compile(r"Average estimated vs inertial difference:\s*([\d.]+)\s*degrees")
COVERAGE_PATTERN = re.compile(r"Estimated orientation coverage:\s*([\d.]+)%")

# CSV file for results
CSV_PATH = RESULTS_DIR / "experiment_results.csv"
CSV_HEADERS = [
  "experiment_name",
  "input_dir",
  "avg_diff_degrees",
  "orientation_coverage_percent",
  "runtime_seconds",
  "arguments",  # Store all arguments as a string for reference
]


def write_result_to_csv(row):
  """Append a single experiment result to the CSV file."""
  file_exists = CSV_PATH.exists()
  with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
    if not file_exists:
      writer.writeheader()
    writer.writerow(row)


def run_experiment(input_dir, **kwargs):
  """Run a single experiment and collect results."""
  # Generate experiment name based on key parameters
  name_parts = [Path(input_dir).name]

  # Add key parameters to experiment name
  if kwargs.get("use_inertials"):
    name_parts.append("inertials")
  else:
    if "window_size" in kwargs:
      name_parts.append(f"w{kwargs['window_size']}")
    if "window_strategy" in kwargs:
      name_parts.append(kwargs["window_strategy"])

  if kwargs.get("relocalize"):
    name_parts.append("relocalize")
  if kwargs.get("produce_360"):
    name_parts.append("360")
  if "output_scale" in kwargs:
    name_parts.append(f"scale{kwargs['output_scale']}")
  if "input_frame_interval" in kwargs:
    name_parts.append(f"interval{kwargs['input_frame_interval']}")
  if "input_frame_scale" in kwargs:
    name_parts.append(f"inscale{kwargs['input_frame_scale']}")
  if "start_frame" in kwargs:
    name_parts.append(f"start{kwargs['start_frame']}")
  if "end_frame" in kwargs:
    name_parts.append(f"end{kwargs['end_frame']}")

  experiment_name = "_".join(name_parts)
  experiment_result_dir = RESULTS_DIR / experiment_name
  experiment_result_dir.mkdir(exist_ok=True)

  # Build command with all arguments from kwargs
  cmd = [
    "python",
    VIDEO_TO_360_PATH,
    input_dir,
  ]

  # Always add produce_debug unless explicitly disabled
  if kwargs.get("produce_debug", True):
    cmd.append("--produce_debug")

  # Add all other arguments
  for key, value in kwargs.items():
    if key == "produce_debug":
      continue  # Already handled

    arg_name = f"--{key}"

    # Boolean flags
    if isinstance(value, bool):
      if value:
        cmd.append(arg_name)
    # Arguments with values
    else:
      cmd.extend([arg_name, str(value)])

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
    "avg_diff_degrees": avg_diff,
    "orientation_coverage_percent": coverage,
    "runtime_seconds": round(runtime, 2),
    "arguments": " ".join(cmd[3:]),  # Store all arguments (excluding python, script, and input_dir)
  }

  # Write result immediately
  write_result_to_csv(result_row)

  return result_row

def main():
  # Collect all experiment configurations
  experiments_to_run = []
  for input_dir in INPUT_DIRS:
    for exp in EXPERIMENTS:
      experiments_to_run.append(
        {
          "input_dir": input_dir,
          **exp,  # Spread the experiment config as kwargs
        }
      )

  print(f"Running {len(experiments_to_run)} experiments with max {PARALLEL_EXPERIMENTS} in parallel...\n")

  # Run experiments in parallel
  with ProcessPoolExecutor(max_workers=PARALLEL_EXPERIMENTS) as executor:
    # Submit all experiments
    future_to_exp = {
      executor.submit(
        run_experiment,
        exp["input_dir"],
        **{k: v for k, v in exp.items() if k != "input_dir"},  # Pass all args except input_dir as kwargs
      ): exp
      for exp in experiments_to_run
    }

    # Collect results as they complete
    for future in as_completed(future_to_exp):
      exp = future_to_exp[future]
      try:
        future.result()
      except Exception as exc:
        print(f"❌ Experiment generated an exception: {exc}")

  print(f"\n✅ All experiments complete! Results saved incrementally to {CSV_PATH}")

if __name__ == "__main__":
  main()
