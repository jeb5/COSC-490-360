import cv2 as cv
from remapping360 import remapping360_torch, remapping360
import helpers
import csv
import math
import torch


def main():
  torch.set_printoptions(precision=3, sci_mode=False)
  input_video_path = "../input/bscr3.mkv"
  output_video_path = "../output/output_360_3.mp4"
  video = cv.VideoCapture(input_video_path)

  cam_matrix, cam_distortion = helpers.BLENDER_CAMERA

  (h, w) = video.get(cv.CAP_PROP_FRAME_HEIGHT), video.get(cv.CAP_PROP_FRAME_WIDTH)

  cam_ideal_center_matrix = cam_matrix.copy()
  cam_ideal_center_matrix[0][2] = w / 2
  cam_ideal_center_matrix[1][2] = h / 2

  vertical_focal_length = cam_matrix[1][1]
  horizontal_focal_length = cam_matrix[0][0]
  vertical_fov = 2 * \
      math.atan(h / (2 * vertical_focal_length)) * 180 / torch.pi
  horizontal_fov = 2 * \
      math.atan(w / (2 * horizontal_focal_length)) * 180 / torch.pi
  # The above is seemingly correct, inline with blender

  output_height = int(h * (180 / vertical_fov))
  output_height = int(output_height * 0.5)
  # output_width = int(w * (360 / horizontal_fov))
  output_width = output_height * 2
  print(
    f"Vertical FOV: {vertical_fov:.2f} degrees, Horizontal FOV: {horizontal_fov:.2f} degrees")
  print(f"Output height: {output_height}, Output width: {output_width}")

  output_video = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(
    *'mp4v'), 10, (output_width, output_height))

  # m1, m2 = cv.fisheye.initUndistortRectifyMap(
  #   cam_matrix, cam_distortion, None, cam_ideal_center_matrix, (w, h), cv.CV_32FC1)
  # image1 = cv.remap(image1, m1, m2, interpolation=cv.INTER_LINEAR,
  #                   borderMode=cv.BORDER_CONSTANT)
  # cv.imwrite('../output/360/image_remapped.png', image1)

  device = torch.accelerator.current_accelerator(
  ).type if torch.accelerator.is_available() else 'cpu'
  if device == 'cpu':
    print("Warning: Using CPU for remapping360_torch, which may be slow")

  with open('../input/bscr3.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
      frame, pitch, yaw, roll = map(float, row)
      degMult = 180 / torch.pi
      pitchdeg, rolldeg, yawdeg = pitch * degMult, roll * degMult, yaw * degMult
      print(f"Frame: {frame}, Pitch: {pitchdeg:.2f}, Roll: {rolldeg:.2f}, Yaw: {yawdeg:.2f}")

      video.set(cv.CAP_PROP_POS_FRAMES, frame)
      ret, image1 = video.read()

      mapX, mapY = remapping360_torch(output_width, output_height, w, h, yaw, pitch, roll, horizontal_focal_length, device)
      # mapX, mapY = remapping360(output_width, output_height, w, h, yaw, pitch, roll, horizontal_focal_length)
      dst = cv.remap(image1, mapX, mapY, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
      output_video.write(dst)
      # cv.imwrite( f'../output/frames/frame_{int(frame)}.png', dst)
  video.release()
  output_video.release()


if __name__ == "__main__":
  main()
