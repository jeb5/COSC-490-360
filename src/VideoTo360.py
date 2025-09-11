import argparse
import math
import numpy as np
from DataManager import DataManager
import cv2 as cv
import torch
from FeatureManager import FeatureManager
import helpers
from helpers import ProcessContext
import remap
import remap_360
import progressbar as pb
import orientation_estimation
from orientation_estimation import (
  rotation_chaining,
  sliding_window,
  overlapping_windows,
)
from ObservationManager import ObservationManager


def main(args):
  # figure = helpers.generate_rotation_histories_plot(
  #   [
  #     {
  #       "name": "Sphere",
  #       "colour": "#42a7f5",
  #       "data": [rot.as_matrix() for (coords, rot) in helpers.generate_fibonacci_sphere_points(300)],
  #       # "vectors": np.array([coords for (coords, rot) in orientation_estimation.generate_fibonacci_sphere_points(300)]),
  #     },
  #   ],
  #   interactive=True,
  #   scatter=True,
  # )
  # return
  dm = DataManager(args.directory, args.input_frame_interval, args.input_frame_scale, args.start_frame, args.end_frame)

  feature_manager = FeatureManager(dm, "SIFT", 0.75, True, True, False)
  intrinsic_matrix, _, _ = dm.get_camera_info()

  def orientation_estimation_func(matches):
    return orientation_estimation.estimate_orientation_change(matches, intrinsic_matrix)

  observation_manager = ObservationManager(
    feature_manager,
    orientation_estimation.is_valid_estimation,
    orientation_estimation_func,
    dm,
  )

  estimated_orientations = (
    dm.get_inertials() if args.use_inertials else estimate_orientations(dm, args, feature_manager, observation_manager)
  )

  if not args.use_inertials:
    dm.write_orientations(estimated_orientations)

  if args.produce_debug and not args.use_inertials:
    output_estimation_information(dm, estimated_orientations, observation_manager, args.interactive)

  if args.produce_360:
    generate_equirectangular_video(dm, estimated_orientations, args.output_scale)


def estimate_orientations(dm, args, feature_manager, observation_manager):
  orientations = None
  if args.window_size == 1:
    print("Using rotation chaining for orientation estimation.")
    orientations = rotation_chaining(dm, feature_manager, args.produce_debug)
  elif args.window_strategy == "simple":
    print("Using simple sliding window for orientation estimation.")
    orientations = sliding_window(dm, args.window_size, observation_manager)
  elif args.window_strategy == "quadratic":
    print("Using sliding window with quadratic lookback for orientation estimation.")
    orientations = sliding_window(dm, args.window_size, observation_manager, quadratic=True)
  elif args.window_strategy == "overlapping":
    print("Using sliding window with overlapping frames for orientation estimation.")
    orientations = overlapping_windows(dm, args.window_size, observation_manager)
  if args.produce_debug:
    dm.save_debug_video()
  # orientations = thing_that_works(dm, feature_manager)
  # orientations = thing_that_does_not_work(dm, feature_manager)
  return orientations


def output_estimation_information(dm, estimated_orientations, observation_manager, interactive):
  sum_deg_difference, count = 0.0, 0.0
  for i in range(len(estimated_orientations)):
    if estimated_orientations[i] is not None:
      sum_deg_difference += (estimated_orientations[i].inv() * dm.get_inertial(i)).magnitude() * (180 / math.pi)
      count += 1
  print(f"Average estimated vs inertial difference: {sum_deg_difference / count:.2f} degrees")
  print(f"Estimated orientation coverage: {100 * count / dm.get_sequence_length():.2f}%")
  comparison_figure = helpers.generate_rotation_histories_plot(
    [
      {
        "name": "Estimated",
        "colour": "#42a7f5",
        "data": [None if rot is None else rot.as_matrix() for rot in estimated_orientations],
      },
      {"name": "Inertial", "colour": "#f07d0a", "data": [rot.as_matrix() for rot in dm.get_inertials()]},
    ],
    interactive=interactive,
  )
  dm.save_image(comparison_figure, "comparison_figure.png")
  dm.save_image(observation_manager.generate_observation_image(), "observations.png")
  if interactive:
    observation_manager.show_interactive_observation_image()


def generate_equirectangular_video(dm, orientations, output_scale):
  device = helpers.get_device()
  intrinsic_matrix, _, (w, h) = dm.get_camera_info()
  focal_length = float(intrinsic_matrix[0][0] + intrinsic_matrix[1][1]) / 2
  vertical_fov = 2 * math.atan(h / (2 * focal_length)) * 180 / math.pi
  out_h = int(h * (180 / vertical_fov) * output_scale)
  out_h = out_h // 2 * 2  # Ensure even height
  out_w = out_h * 2

  background = torch.zeros((out_h, out_w, 4), dtype=torch.float32, device=device)
  output_vectors = remap_360.getFrameOutputVectors(out_w, out_h, device)

  N = 0
  last_N_frames = []

  prefix_widgets = ["Generating 360 video | Frame ", pb.Counter(format="%(value)d"), "/", pb.FormatLabel("%(max_value)d")]
  with ProcessContext(prefix_widgets=prefix_widgets, max_value=len(orientations) - 1) as bar:
    for frame_number in bar(range(len(orientations))):
      rotation = orientations[frame_number]
      frame = dm.get_frame(frame_number, undistort=True)

      frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
      frame = torch.from_numpy(frame).to(device).float()

      if rotation is not None:
        map360 = remap_360.remapping360_torch(w, h, rotation, focal_length, output_vectors)
        dst = remap.torch_remap(map360, frame)
        background = helpers.add_transparent_image_torch(background, dst)

      last_N_frames.append(background)
      if frame_number > N:
        last_N_frames.pop(0)
      output_frame = torch.mean(torch.stack(last_N_frames), dim=0)
      dm.write_360_frame(output_frame)
  dm.save_360_video()
  print("Done.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process video+inertials to produce 360 video")
  parser.add_argument("directory", type=str, help="Path to directory containing video+inertials")
  parser.add_argument("--produce_debug", action="store_true")
  parser.add_argument("--interactive", action="store_true")
  parser.add_argument("--produce_360", action="store_true")
  # parser.add_argument("--use_features_cache", action="store_true")
  # parser.add_argument("--use_matches_cache", action="store_true")
  parser.add_argument("--use_inertials", action="store_true")
  parser.add_argument("--window_size", type=int, default=1)
  parser.add_argument("--window_strategy", type=str, choices=["simple", "quadratic", "overlapping"], default="simple")
  parser.add_argument("--output_scale", type=float, default=1.0)
  parser.add_argument("--input_frame_interval", type=int, default=1)
  parser.add_argument("--input_frame_scale", type=float, default=1.0)
  parser.add_argument("--start_frame", type=int, default=0)
  parser.add_argument("--end_frame", type=int, default=None)

  args = parser.parse_args()
  main(args)
