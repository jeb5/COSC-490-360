import argparse
import subprocess
import json


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
    "packet=pts_time,flags",
    "-of",
    "json",
    video_path,
  ]

  result = subprocess.run(cmd, capture_output=True, text=True)
  packets = json.loads(result.stdout).get("packets", [])

  keyframes = []
  for pkt in packets:
    if "K" in pkt.get("flags", ""):
      keyframes.append(float(pkt["pts_time"]))

  if not keyframes:
    raise ValueError(f"No keyframes found near {target_time:.2f}s.")

  if direction == "before":
    return max([t for t in keyframes if t <= target_time], default=keyframes[0])
  elif direction == "after":
    return min([t for t in keyframes if t >= target_time], default=keyframes[-1])
  else:
    raise ValueError("Direction must be 'before' or 'after'")


def trim_video(input_path, output_path, start_time, end_time):
  cmd = ["ffmpeg", "-y", "-ss", f"{start_time}", "-to", f"{end_time}", "-i", input_path, "-c", "copy", output_path]
  subprocess.run(cmd)


def main(args):
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
    args.input,
  ]
  out = subprocess.run(cmd, capture_output=True, text=True)
  rate = out.stdout.strip()
  num, denom = map(int, rate.split("/"))
  fps = num / denom

  print(f"[INFO] Using FPS: {fps:.3f}")

  start_kf = get_keyframe_near(args.input, args.start_frame / fps, direction="before")
  end_kf = get_keyframe_near(args.input, args.end_frame / fps, direction="after")

  start_kf_frame = int(start_kf * fps)
  end_kf_frame = int(end_kf * fps)

  print(f"[INFO] Requested frames: {args.start_frame} to {args.end_frame}")
  print(f"[INFO] Nearest keyframes: {start_kf:.3f}s to {end_kf:.3f}s")

  trim_video(args.input, args.output, start_kf, end_kf)

  print(f"[DONE] Trimmed video saved to {args.output}")
  print(f"[INFO] Actual start frame: {start_kf_frame}, end frame: {end_kf_frame}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Trim video on nearest keyframes using ffmpeg -c copy.")
  parser.add_argument("input", help="Path to input video")
  parser.add_argument("output", help="Path to output video")
  parser.add_argument("start_frame", type=int, help="Desired start frame")
  parser.add_argument("end_frame", type=int, help="Desired end frame")
  args = parser.parse_args()
  main(args)
