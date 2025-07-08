import os
import signal
import sys
import cv2 as cv
import remap_360
import helpers
from video_writer import VideoWriter
import math
import remap
import torch
import line_profiler
import argparse


@line_profiler.profile
def main(args):
  torch.set_printoptions(precision=3, sci_mode=False)
  input_video_path = helpers.get_file_path_pack_dir(args.input_directory, "video")
  input_rotation_path = helpers.get_file_path_pack_dir(args.input_directory, args.rotation_file_type)
  output_video_path = (
    args.output_path
    if args.output_path
    else os.path.join(args.input_directory, f"{os.path.basename(args.input_directory)}_360.mp4")
  )
  input_video = cv.VideoCapture(input_video_path)

  cam_matrix, cam_distortion = helpers.load_camera_info(helpers.get_file_path_pack_dir(args.input_directory, "camera_info"))

  in_w, in_h = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  input_size = (in_w, in_h)

  vertical_focal_length = cam_matrix[1][1]
  horizontal_focal_length = cam_matrix[0][0]
  vertical_fov = 2 * math.atan(in_h / (2 * vertical_focal_length)) * 180 / torch.pi
  horizontal_fov = 2 * math.atan(in_w / (2 * horizontal_focal_length)) * 180 / torch.pi

  out_h = int(in_h * (180 / vertical_fov) * args.scale)
  out_h = out_h // 2 * 2  # Ensure even height
  out_w = out_h * 2
  print(f"Detected FOV: {horizontal_fov:.4f}˚x{vertical_fov:.4f}˚")
  print(f"Output size: {out_w}x{out_h}")

  input_fps = int(input_video.get(cv.CAP_PROP_FPS))
  output_video = VideoWriter(output_video_path, input_fps, (out_w, out_h), mbps=15, spherical_metadata=True)

  device = helpers.get_device()

  ideal_cam_mat = (
    cv.fisheye.estimateNewCameraMatrixForUndistortRectify(cam_matrix, cam_distortion, input_size, None, None, 1, input_size, 1)
    if len(cam_distortion) > 0
    else cam_matrix.copy()
  )
  ideal_cam_mat[0][2] = in_w / 2
  ideal_cam_mat[1][2] = in_h / 2
  ideal_focal_length = ideal_cam_mat[0][0]
  undistort_map = None
  if len(cam_distortion) > 0:
    m1, m2 = cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, ideal_cam_mat, input_size, cv.CV_32FC1)
    m1, m2 = torch.from_numpy(m1).to(device), torch.from_numpy(m2).to(device)
    undistort_map = remap.absoluteToRelative(torch.stack((m1, m2), dim=-1), input_size)

  output_vectors = remap_360.getFrameOutputVectors(out_w, out_h, device)
  background = torch.zeros((out_h, out_w, 4), dtype=torch.float32, device=device)

  def cleanup():
    input_video.release()
    output_video.save_video()
    print("Done.")

  def interuppt_handler(signum, frame):
    cleanup()
    sys.exit(0)

  signal.signal(signal.SIGINT, interuppt_handler)

  # generator:
  rotations = helpers.rotations_from_csv(input_rotation_path)
  next_rotation = next(rotations, None)

  for frame in range(int(input_video.get(cv.CAP_PROP_FRAME_COUNT))):
    rotation = None
    if next_rotation is not None and next_rotation[0] == frame:
      rotation = next_rotation[1]
      next_rotation = next(rotations, None)

    input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
    ret, image = input_video.read()
    image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
    image = torch.from_numpy(image).to(device).float()
    # image = helpers.apply_circular_vignette_alpha(image, 0.85, 0.9)
    # image = helpers.apply_combined_vignette_alpha(image, 0.75, 0.8, 0.0, 0.05)
    image = helpers.apply_combined_vignette_alpha(image, circ_start_pct=0.7, rect_start_pct=0.1)
    image = remap.torch_remap(undistort_map, image) if undistort_map is not None else image

    if rotation is not None:
      pitch, roll, yaw = rotation
      print(f"Frame: {frame}, Pitch: {pitch:.2f}˚, Roll: {roll:.2f}˚, Yaw: {yaw:.2f}˚")
      map360 = remap_360.remapping360_torch(in_w, in_h, yaw, pitch, roll, ideal_focal_length, output_vectors)
      dst = remap.torch_remap(map360, image)
      background = helpers.add_transparent_image_torch(background, dst)

    output_video.write_frame(background)
  cleanup()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Remap rotating fotoage to 360 video")
  parser.add_argument("input_directory", type=str, help="Path to the input video file.")
  parser.add_argument("rotation_file_type", type=str, choices=["inertial", "visual"], help="Type of rotation data file to use.")
  parser.add_argument("--output_path", type=str, help="Path to the output video file.")
  parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for output video size.")
  args = parser.parse_args()
  with torch.no_grad():
    main(args)
