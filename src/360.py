import cv2 as cv
import remapping360
import helpers
import csv
from VideoWriter import VideoWriter
import math
import remap
import torch
import line_profiler
import argparse


@line_profiler.profile
def main(args):
  torch.set_printoptions(precision=3, sci_mode=False)
  input_video = cv.VideoCapture(args.video_path)

  cam_matrix, cam_distortion = helpers.GOPRO_CAMERA

  in_w, in_h = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  input_size = (in_w, in_h)

  vertical_focal_length = cam_matrix[1][1]
  horizontal_focal_length = cam_matrix[0][0]
  vertical_fov = 2 * math.atan(in_h / (2 * vertical_focal_length)) * 180 / torch.pi
  horizontal_fov = 2 * math.atan(in_w / (2 * horizontal_focal_length)) * 180 / torch.pi

  out_h = int(in_h * (180 / vertical_fov) * args.scale)
  out_w = out_h * 2
  print(f"Vertical FOV: {vertical_fov:.4f} degrees, Horizontal FOV: {horizontal_fov:.4f} degrees")
  print(f"Output height: {out_h}, Output width: {out_w}")

  output_video = VideoWriter(args.output_path, 20, (out_w, out_h), mbps=15, spherical_metadata=True)

  device = helpers.get_device()

  newMat = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(cam_matrix, cam_distortion, input_size, None, None, 1, input_size, 1)
  print(f"New camera matrix: {newMat}")
  newMat[0][2] = in_w / 2
  newMat[1][2] = in_h / 2
  newFocalLength = newMat[0][0]
  m1, m2 = cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, newMat, input_size, cv.CV_32FC1)
  m1, m2 = torch.from_numpy(m1).to(device), torch.from_numpy(m2).to(device)
  undistortMap = torch.stack((m1, m2), dim=-1)
  undistortMap = remap.absoluteToRelative(undistortMap, input_size)

  output_vectors = remapping360.getFrameOutputVectors(out_w, out_h, device)
  background = None
  with open(args.rotation_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    i = 0
    for row in reader:
      i += 1
      if i > 100:
        break
      frame, pitch, yaw, roll = map(float, row)
      yaw += math.pi / 2
      degMult = 180 / torch.pi
      pitchdeg, rolldeg, yawdeg = pitch * degMult, roll * degMult, yaw * degMult
      print(f"Frame: {frame}, Pitch: {pitchdeg:.2f}, Roll: {rolldeg:.2f}, Yaw: {yawdeg:.2f}")

      input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
      ret, image1 = input_video.read()
      image1 = cv.cvtColor(image1, cv.COLOR_BGR2BGRA)
      image1 = torch.from_numpy(image1).to(device).float()
      image1 = remap.torch_remap(undistortMap, image1)

      mapX, mapY = remapping360.remapping360_torch(out_w, out_h, in_w, in_h, yaw, pitch,
                                                   roll, newFocalLength, output_vectors, device)
      mx, my = torch.from_numpy(mapX).to(device), torch.from_numpy(mapY).to(device)
      map360 = torch.stack((mx, my), dim=-1)
      map360 = remap.absoluteToRelative(map360, (in_w, in_h))
      dst = remap.torch_remap(map360, image1)

      if background is None:
        background = dst.clone()
      else:
        background = helpers.add_transparent_image_torch(background, dst)

      output_video.write_frame(background)
  input_video.release()
  output_video.save_video()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Remap rotating fotoage to 360 video")
  parser.add_argument("video_path", type=str, help="Path to the input video file.")
  parser.add_argument("rotation_path", type=str, help="Path to the input rotation file.")
  parser.add_argument("output_path", type=str, help="Path to the output video file.")
  parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for output video size.")
  args = parser.parse_args()
  main(args)
