import csv
import re
import subprocess
import time
from pathlib import Path

# --- Configuration ---
BASE_COMMAND = ["python", "src/angle_estimation.py"]
INPUT_DIRECTORIES = [
  # "input/synthetic1",
  # "input/synthetic2",
  # "input/synthetic3",
  "input/synthetic4",
]
CONFIGURATIONS = [
  (1, "simple"),
  (10, "simple"),
  (10, "quadratic"),
]
OUTPUT_CSV_FILE = "experiment_results.csv"

ANGLE_DIFF_RE = re.compile(r"Average angle difference: ([\d.]+) degrees", re.IGNORECASE)
VISUAL_PRED_RE = re.compile(r"Visual predictions: ([\d.]+)%", re.IGNORECASE)


def run_single_experiment(input_dir, window_size, window_mode):
  print(f"--- Running experiment ---")
  print(f"Directory:     {input_dir}")
  print(f"Window Size:   {window_size}")
  print(f"Window Mode:   {window_mode}")
  print("--------------------------")

  artifacts_dir = Path("artifacts")
  artifacts_dir.mkdir(exist_ok=True)

  input_name = Path(input_dir.rstrip("/")).name
  output_image_name = f"{input_name}_w{window_size}_{window_mode}.png"
  output_image_path = artifacts_dir / output_image_name

  command = [
    *BASE_COMMAND,
    input_dir,
    "--window_size",
    str(window_size),
    "--window_mode",
    window_mode,
    "--use_features_cache",
    "--figure_output",
    str(output_image_path),
  ]

  result_data = {
    "directory": input_dir,
    "window_size": window_size,
    "window_mode": window_mode,
    "execution_time_sec": None,
    "avg_angle_difference": None,
    "visual_prediction_percent": None,
    "error": None,
  }

  try:
    start_time = time.time()
    process_result = subprocess.run(command, capture_output=True, text=True, check=True, encoding="utf-8")
    end_time = time.time()
    result_data["execution_time_sec"] = round(end_time - start_time, 2)

    stdout = process_result.stdout
    print(f"Script output captured. Execution time: {result_data['execution_time_sec']}s")

    angle_match = ANGLE_DIFF_RE.search(stdout)
    visual_match = VISUAL_PRED_RE.search(stdout)

    if angle_match:
      result_data["avg_angle_difference"] = float(angle_match.group(1))
      print(f"  > Found Angle Difference: {result_data['avg_angle_difference']}")
    else:
      print("  > WARNING: Could not find 'Average angle difference' in output.")
      result_data["error"] = "Angle difference not found"

    if visual_match:
      result_data["visual_prediction_percent"] = float(visual_match.group(1))
      print(f"  > Found Visual Predictions: {result_data['visual_prediction_percent']}%")
    else:
      print("  > WARNING: Could not find 'Visual predictions' in output.")
      result_data["error"] = (
        result_data["error"] + "; Visual prediction not found" if result_data["error"] else "Visual prediction not found"
      )

  except FileNotFoundError:
    error_msg = f"Error: The script '{' '.join(BASE_COMMAND)}' was not found."
    print(error_msg)
    result_data["error"] = error_msg
  except subprocess.CalledProcessError as e:
    error_msg = f"Error executing script. It returned a non-zero exit code: {e.returncode}\nStderr:\n{e.stderr}"
    print(error_msg)
    result_data["error"] = error_msg
  except Exception as e:
    error_msg = f"An unexpected error occurred: {e}"
    print(error_msg)
    result_data["error"] = error_msg

  print("\n")
  return result_data


def write_result_to_csv(headers, result, file_path):
  """Writes a single result row to the CSV file."""
  try:
    file_exists = Path(file_path).exists()
    with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=headers)
      if not file_exists:
        writer.writeheader()
      writer.writerow(result)
  except IOError as e:
    print(f"Error writing to CSV file '{file_path}': {e}")


def main():
  print(f"Starting experiments. Results will be saved to '{OUTPUT_CSV_FILE}'")

  csv_headers = [
    "directory",
    "window_size",
    "window_mode",
    "execution_time_sec",
    "avg_angle_difference",
    "visual_prediction_percent",
    "error",
  ]

  for directory in INPUT_DIRECTORIES:
    if not Path(directory).is_dir():
      print(f"WARNING: Directory '{directory}' not found. Skipping.")
      continue

    print(f"Performing warm-up for '{directory}'... (To pre-populate cache)")
    warmup_command = [
      *BASE_COMMAND,
      directory,
      "--warmup",
    ]
    try:
      subprocess.run(warmup_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      print("  > Warm-up completed successfully.")
    except Exception as e:
      print(f"  > Warm-up failed: {e}")

    for size, mode in CONFIGURATIONS:
      result = run_single_experiment(directory, size, mode)
      if result:
        write_result_to_csv(csv_headers, result, OUTPUT_CSV_FILE)

  print(f"All experiments complete. Results saved to '{OUTPUT_CSV_FILE}'.")


if __name__ == "__main__":
  main()
