import csv
import argparse


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


def extract_pitch_yaw_roll(input_csv, output_csv, start_frame, end_frame, reduce_method="average"):
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
  extract_pitch_yaw_roll(
    input_csv=args.input_csv,
    output_csv=args.output_csv,
    start_frame=args.start_frame,
    end_frame=args.end_frame,
    reduce_method=args.reduce_method,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Extract pitch, yaw, and roll from a CSV file.")
  parser.add_argument("input_csv", type=str, help="Path to the input CSV file.")
  parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")
  parser.add_argument("--start_frame", type=int, default=0, help="Start frame number.")
  parser.add_argument("--end_frame", type=int, default=1000, help="End frame number.")
  parser.add_argument(
    "--reduce_method",
    type=str,
    choices=["average", "last", "middle"],
    default="middle",
    help="Method to reduce multiple rows for the same frame.",
  )

  args = parser.parse_args()
  main(args)
