import argparse
import os
import subprocess
import csv
import json

import helpers


def get_keyframe_near(video_path, target_time, direction="before", window=5):
  """
  Returns the timestamp of the nearest keyframe to `target_time`,
  within a window of `±window` seconds.
  """
  # Only look around the target_time ± window
  start_time = max(target_time - window, 0)
  cmd = [
    "ffprobe",
    "-v",
    "error",
    "-read_intervals",
    f"{start_time}%+{2 * window}",
    "-select_streams",
    "v",
    "-show_entries",
    "frame=pts_time,pict_type",
    "-of",
    "json",
    video_path,
  ]

  result = subprocess.run(cmd, capture_output=True, text=True)
  frames = json.loads(result.stdout).get("frames", [])

  keyframes = []
  for frame in frames:
    if frame.get("pict_type") == "I":
      keyframes.append(float(frame["pts_time"]))

  if not keyframes:
    raise ValueError(f"No I-frames found near {target_time:.2f}s.")

  if direction == "before":
    return max([t for t in keyframes if t <= target_time], default=keyframes[0])
  elif direction == "after":
    return min([t for t in keyframes if t >= target_time], default=keyframes[-1])
  else:
    raise ValueError("Direction must be 'before' or 'after'")


def trim_video(input_path, output_path, start_time, end_time):
  cmd = [
    "ffmpeg",
    "-hide_banner",
    "-fflags",
    "+genpts+igndts",
    "-y",
    "-ss",
    f"{start_time}",
    "-to",
    f"{end_time}",
    "-i",
    input_path,
    "-c",
    "copy",
    output_path,
  ]
  subprocess.run(cmd)

def parse_header_get_indices(header):
  """Returns column indices for frame, pitch, yaw, roll"""
  header = [h.strip() for h in header]
  return {
    "frame": header.index("frame"),
    "pitch": header.index("org_pitch"),
    "yaw": header.index("org_roll"),
    "roll": header.index("org_yaw"),
    # Yaw and roll are swapped in the gopro CSV
  }


def binary_search_first_frame(filename, target_frame):
  with open(filename, "rb") as f:
    f.seek(0, 2)
    file_size = f.tell()
    low = 0
    high = file_size
    best_pos = None

    while low <= high:
      mid = (low + high) // 2
      f.seek(mid)
      if mid != 0:
        f.readline()  # skip partial line

      pos = f.tell()
      line = f.readline()
      if not line:
        break
      try:
        frame = int(line.decode("utf-8").split(",")[0])
      except ValueError:
        break

      if frame < target_frame:
        low = f.tell()
      else:
        best_pos = pos
        high = mid - 1

    return best_pos


def reduce_rows(rows, method):
  if not rows:
    return (0.0, 0.0, 0.0)

  if method == "average":
    n = len(rows)
    result = tuple(sum(vals[i] for vals in rows) / n for i in range(3))
  elif method == "last":
    result = rows[-1]
  elif method == "middle":
    result = rows[len(rows) // 2]
  else:
    raise ValueError("Unknown method: " + method)

  result = (result[0] - 90, result[1], result[2])
  return tuple(round(x, 3) for x in result)


def extract_pitch_yaw_roll(input_csv, output_csv, start_frame, end_frame, reduce_method="middle"):
  with open(input_csv, "rb") as f:
    # Parse header
    header_line = f.readline()
    header = header_line.decode("utf-8").strip().split(",")
    idx = parse_header_get_indices(header)

    # Binary search to the first relevant frame
    pos = binary_search_first_frame(input_csv, start_frame)
    if pos is None:
      print(f"Frame {start_frame} not found.")
      return

    f.seek(pos)

    current_frame = None
    current_block = []
    output_rows = []
    output_frame_number = 0

    for line in f:
      row = line.decode("utf-8").strip().split(",")
      try:
        frame = int(row[idx["frame"]])
      except ValueError:
        continue  # skip malformed lines

      if frame > end_frame:
        break
      if frame < start_frame:
        continue

      try:
        pitch = float(row[idx["pitch"]])
        yaw = float(row[idx["yaw"]])
        roll = float(row[idx["roll"]])
      except ValueError:
        continue  # skip malformed data rows

      if current_frame is None:
        current_frame = frame

      if frame == current_frame:
        current_block.append((pitch, yaw, roll))
      else:
        # Process completed block
        pitch, yaw, roll = reduce_rows(current_block, reduce_method)
        output_rows.append([output_frame_number, pitch, roll, yaw])
        output_frame_number += 1

        # Start new block
        current_frame = frame
        current_block = [(pitch, yaw, roll)]

    # Process last block
    if current_block and start_frame <= current_frame <= end_frame:
      pitch, yaw, roll = reduce_rows(current_block, reduce_method)
      output_rows.append([output_frame_number, pitch, roll, yaw])

  with open(output_csv, "w", newline="") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["frame", "pitch", "roll", "yaw"])
    writer.writerows(output_rows)

def main(args):
  if args.reduce_method == "average":
    print("Warning: 'Average' reduce method will lead to errors if the angle wraps around. Currently unusable")

  input_csv_path = helpers.get_file_path_pack_dir(args.input, "inertial")
  input_video_path = helpers.get_file_path_pack_dir(args.input, "video")
  output_csv_path = helpers.get_file_path_pack_dir(args.output, "inertial")
  output_video_path = helpers.get_file_path_pack_dir(args.output, "video")

  # Try to detect FPS
  cmd = [
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=r_frame_rate",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    input_video_path,
  ]
  out = subprocess.run(cmd, capture_output=True, text=True)
  rate = out.stdout.strip()
  num, denom = map(int, rate.split("/"))
  fps = num / denom

  print(f"[INFO] Using FPS: {fps:.3f}")

  start_kf = get_keyframe_near(input_video_path, args.start_frame / fps, direction="before")
  end_kf = get_keyframe_near(input_video_path, args.end_frame / fps, direction="after")

  start_kf_frame = int(start_kf * fps)
  end_kf_frame = int(end_kf * fps) + 1
  end_kf = end_kf_frame / fps

  print(f"[INFO] Requested frames: {args.start_frame} to {args.end_frame}")
  print(f"[INFO] Actual start frame: {start_kf_frame}, end frame: {end_kf_frame} (based on nearest keyframes)")

  trim_video(input_video_path, output_video_path, start_kf, end_kf)

  print(f"[DONE] Trimmed video saved to {args.output}")

  extract_pitch_yaw_roll(
    input_csv=input_csv_path,
    output_csv=output_csv_path,
    start_frame=start_kf_frame,
    end_frame=end_kf_frame,
    reduce_method=args.reduce_method,
  )

  print(f"[DONE] Processed inertial data saved to {output_csv_path}")

  caminfo_path = helpers.get_file_path_pack_dir(args.input, "camera_info")
  if caminfo_path:
    # copy caminfo file to output directory
    output_caminfo_path = os.path.join(args.output, os.path.basename(caminfo_path))
    with open(caminfo_path, "r") as f:
      caminfo_data = f.read()
    with open(output_caminfo_path, "w") as f:
      f.write(caminfo_data)
    print(f"[DONE] Camera info copied to {output_caminfo_path}")
  else:
    print("[WARNING] No camera info file found in input directory.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Trim video on nearest keyframes using ffmpeg -c copy, and extract relevant rotations from CSV. Currently specifically designed for GoPro inertial data CSV files."
  )
  parser.add_argument("input", help="Path to input directory containing full-length video and inertial CSV file")
  parser.add_argument("output", help="Path to output directory for trimmed video and processed CSV file")
  parser.add_argument("start_frame", type=int, help="Desired start frame")
  parser.add_argument("end_frame", type=int, help="Desired end frame")
  parser.add_argument(
    "--reduce_method",
    type=str,
    choices=["average", "last", "middle"],
    default="middle",
    help="Method to reduce multiple rows for the same frame.",
  )
  args = parser.parse_args()
  main(args)
