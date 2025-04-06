import cv2 as cv
import remapping360
import STMaps
import helpers
import csv
import os
import math
import torch


def main():
  torch.set_printoptions(precision=3, sci_mode=False)
  input_video_path = "./input/bscr7.mkv"
  input_rotation_path = "./input/bscr5.csv"
  output_video_path = "./output/output_360_5.mp4"
  video = cv.VideoCapture(input_video_path)

  cam_matrix, cam_distortion = helpers.BLENDER_CAMERA_2

  h, w = int(video.get(cv.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv.CAP_PROP_FRAME_WIDTH))
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
  output_height = int(output_height / 8)
  # output_height = 760
  # output_width = int(w * (360 / horizontal_fov))
  output_width = output_height * 2
  print(f"Vertical FOV: {vertical_fov:.4f} degrees, Horizontal FOV: {horizontal_fov:.4f} degrees")
  print(f"Output height: {output_height}, Output width: {output_width}")

  temp_output_path = "/tmp/360_output.mp4"
  output_video = cv.VideoWriter(temp_output_path, cv.VideoWriter_fourcc(
    *'mp4v'), 20, (output_width, output_height))


  device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
  if device == 'cpu':
    print("Warning: Using CPU for remapping360_torch, which may be slow")

  # m1, m2 = cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, cam_ideal_center_matrix, size, cv.CV_32FC1)
  # m1, m2 = torch.from_numpy(m1).to(device), torch.from_numpy(m2).to(device)
  # undistortMap = torch.stack((m1, m2), dim=-1)
  bcpfuMap = helpers.blenderCyclesPolynomialFisheyeUndistort(size, [])
  bcpfuMap = bcpfuMap.to(device)

  output_vectors = remapping360.getFrameOutputVectors(output_width, output_height, device)
  background = None
  # ground_truth = cv.imread("./input/golden_gate_hills.png")
  # ground_truth = cv.cvtColor(ground_truth, cv.COLOR_BGR2BGRA)
  with open(input_rotation_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
      frame, pitch, yaw, roll = map(float, row)
      yaw += math.pi / 2
      degMult = 180 / torch.pi
      pitchdeg, rolldeg, yawdeg = pitch * degMult, roll * degMult, yaw * degMult
      print(f"Frame: {frame}, Pitch: {pitchdeg:.2f}, Roll: {rolldeg:.2f}, Yaw: {yawdeg:.2f}")

      video.set(cv.CAP_PROP_POS_FRAMES, frame)
      ret, image1 = video.read()
      image1 = cv.cvtColor(image1, cv.COLOR_BGR2BGRA)
      image1 = torch.from_numpy(image1).to(device)
      # image1 = cv.remap(image1, m1, m2, interpolation=cv.INTER_LINEAR,
      #                   borderMode=cv.BORDER_CONSTANT)
      image1 = helpers.torch_remap(bcpfuMap, image1)
      cv.imshow("Image1", image1.cpu().numpy())
      cv.waitKey(0)

      mapX, mapY = remapping360.remapping360_torch(output_width, output_height, w, h, yaw, pitch,
                                                   roll, horizontal_focal_length, output_vectors, device)
      # Temporary
      mx, my = torch.from_numpy(mapX).to(device), torch.from_numpy(mapY).to(device)
      map360 = torch.stack((mx, my), dim=-1)
      # mapX, mapY = remapping360(output_width, output_height, w, h, yaw, pitch, roll, horizontal_focal_length)
      # dst = cv.remap(image1, mapX, mapY, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
      # combined_map = helpers.combineRemappings(undistortMap, map360)
      combined_map = map360
      dst = helpers.torch_remap(combined_map, image1)

      dst = dst.cpu().numpy()
      # image1 = image1.cpu().numpy()
      # dst = STMaps.remapTransborder(image1, mapX, mapY)
      if background is None:
        background = dst.copy()
      else:
        # print background type
        # print(f"Background color type: {background.dtype}")
        # cv.imshow("Background", background)
        # cv.waitKey(0)
        # background = helpers.add_transparent_image(ground_truth, dst)
        background = helpers.add_transparent_image(background, dst)
      # cv.imshow("dst", dst)
      # cv.waitKey(0)

      output_video.write(helpers.BGRAToBGRAlphaBlack(background))
  video.release()
  output_video.release()
  helpers.addSphericalMetadata(temp_output_path, output_video_path)
  os.remove(temp_output_path)

if __name__ == "__main__":
  main()
