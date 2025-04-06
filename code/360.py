import cv2 as cv
import remapping360
import helpers
import csv
from VideoWriter import VideoWriter
import math
import remap
import torch
import numpy as np


def main():
  torch.set_printoptions(precision=3, sci_mode=False)
  input_video_path = "./output/remapped_bscr9.mp4"
  input_rotation_path = "./input/bscr5.csv"
  output_video_path = "./output/output_360_5.mp4"
  input_video = cv.VideoCapture(input_video_path)

  cam_matrix, cam_distortion = helpers.BLENDER_CAMERA_2

  h, w = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT)), int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
  size = (w, h)

  cam_ideal_center_matrix = cam_matrix.copy()
  cam_ideal_center_matrix[0][2] = w / 2
  cam_ideal_center_matrix[1][2] = h / 2

  vertical_focal_length = cam_matrix[1][1]
  horizontal_focal_length = cam_matrix[0][0]
  vertical_fov = 2 * math.atan(h / (2 * vertical_focal_length)) * 180 / torch.pi
  horizontal_fov = 2 * math.atan(w / (2 * horizontal_focal_length)) * 180 / torch.pi

  output_height = int(h * (180 / vertical_fov))
  # TODO: Make code run faster enough to not need this
  output_height = int(output_height / 4)
  # output_height = 760
  # output_width = int(w * (360 / horizontal_fov))
  output_width = output_height * 2
  print(f"Vertical FOV: {vertical_fov:.4f} degrees, Horizontal FOV: {horizontal_fov:.4f} degrees")
  print(f"Output height: {output_height}, Output width: {output_width}")

  output_video = VideoWriter(output_video_path, 20, (output_width, output_height), mbps=15, spherical_metadata=True)

  device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
  if device == 'cpu':
    print("Warning: Using CPU for remapping, which may be slow")

  newMat = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(cam_matrix, cam_distortion, size, None, None, 1, size, 1)
  newFocalLength = newMat[0][0]
  m1, m2 = cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, newMat, size, cv.CV_32FC1)
  m1, m2 = torch.from_numpy(m1).to(device), torch.from_numpy(m2).to(device)
  undistortMap = torch.stack((m1, m2), dim=-1)
  undistortMap = remap.absoluteToRelative(undistortMap, size)

  output_vectors = remapping360.getFrameOutputVectors(output_width, output_height, device)
  background = None
  with open(input_rotation_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
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

      mapX, mapY = remapping360.remapping360_torch(output_width, output_height, w, h, yaw, pitch,
                                                   roll, newFocalLength, output_vectors, device)
      mx, my = torch.from_numpy(mapX).to(device), torch.from_numpy(mapY).to(device)
      map360 = torch.stack((mx, my), dim=-1)
      map360 = remap.absoluteToRelative(map360, (w, h))
      dst = remap.torch_remap(map360, image1)

      if background is None:
        background = dst.clone()
      else:
        background = helpers.add_transparent_image_torch(background, dst)

      output_video.write_frame(background)
  input_video.release()
  output_video.save_video()

if __name__ == "__main__":
  main()
