import argparse
import math
from DataManager import DataManager
import cv2 as cv
import torch

import helpers
from helpers import ProcessContext
import remap
import remap_360
import progressbar as pb


def main(args):
  dm = DataManager(args.directory, args.input_frame_interval, args.input_frame_scale)

  estimated_orientations = dm.get_inertials() if args.use_inertials else estimate_orientations(dm, args)

  if args.produce_360:
    generate_equirectangular_video(dm, estimated_orientations, args)


def estimate_orientations(dm, args):
  pass


def generate_equirectangular_video(dm, orientations, args):
  device = helpers.get_device()
  intrinsic_matrix, _, (w, h) = dm.get_camera_info()
  focal_length = float(intrinsic_matrix[0][0] + intrinsic_matrix[1][1]) / 2
  vertical_fov = 2 * math.atan(h / (2 * focal_length)) * 180 / math.pi
  out_h = int(h * (180 / vertical_fov) * args.output_scale)
  out_h = out_h // 2 * 2  # Ensure even height
  out_w = out_h * 2

  background = torch.zeros((out_h, out_w, 4), dtype=torch.float32, device=device)
  output_vectors = remap_360.getFrameOutputVectors(out_w, out_h, device)

  N = 1
  last_N_frames = []

  prefix_widgets = ["Generating 360 video | Frame ", pb.Counter(format="%(value)d"), "/", pb.FormatLabel("%(max_value)d")]
  with ProcessContext(prefix_widgets=prefix_widgets, max_value=len(orientations) - 1) as bar:
    print()
    for frame_number in bar(range(len(orientations))):
      rotation = orientations[frame_number]
      frame = dm.get_frame(frame_number, undistort=True)

      frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
      frame = torch.from_numpy(frame).to(device).float()

      if rotation is not None:
        yaw, pitch, roll = rotation.as_euler("ZXY")
        map360 = remap_360.remapping360_torch(w, h, yaw, pitch, roll, focal_length, output_vectors)
        dst = remap.torch_remap(map360, frame)
        background = helpers.add_transparent_image_torch(background, dst)

      last_N_frames.append(background)
      if frame_number > N:
        last_N_frames.pop(0)
      output_frame = torch.mean(torch.stack(last_N_frames), dim=0)
      dm.write_360_frame(output_frame)
  dm.save_360_video()
  print()
  print("Done.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process video+inertials to produce 360 video")
  parser.add_argument("directory", type=str, help="Path to directory containing video+inertials")
  parser.add_argument("--produce_debug", action="store_true")
  parser.add_argument("--produce_360", action="store_true")
  # parser.add_argument("--use_features_cache", action="store_true")
  # parser.add_argument("--use_matches_cache", action="store_true")
  parser.add_argument("--use_inertials", action="store_true")
  parser.add_argument("--window_size", type=int, default=1)
  parser.add_argument("--window_strategy", type=str, choices=["simple", "quadratic", "overlapping"], default="simple")
  parser.add_argument("--output_scale", type=float, default=1.0)
  parser.add_argument("--input_frame_interval", type=int, default=1)
  parser.add_argument("--input_frame_scale", type=float, default=1.0)

  args = parser.parse_args()
  main(args)
