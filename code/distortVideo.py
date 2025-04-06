import cv2 as cv
import remap
import helpers
import torch
import numpy as np
import code.VideoWriter as VideoWriter


def main():
  video_path = "./input/bscr9.mkv"
  output_path = "./output/remapped_bscr9.mp4"
  input_video = cv.VideoCapture(video_path)
  if not input_video.isOpened():
    print("Error: Could not open video.")
    return
  cam_mat, cam_distortion = helpers.BLENDER_CAMERA_3

  width, height = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  output_width, output_height = int(width / 1.5), int(height / 1.5)
  # horizontal_focal_length = cam_mat[0][0]
  # horizontal_fov_times_1p5 = 2 * math.atan(width * 1.5 / (2 * horizontal_focal_length)) * 180 / torch.pi
  # print(f"Horizontal FOV x 1.5: {horizontal_fov_times_1p5:.4f} degrees")

  output_video = VideoWriter.VideoWriter(output_path, 20, (output_width, output_height), mbps=10)

  device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
  if device == 'cpu':
    print("Warning: Using CPU remapping, which may be slow")

  distortion_map = remap.getFisheyeDistortionMap((width, height), cam_mat, cam_distortion).to(device)

  frameNumber = 0
  while True:
    ret, frame = input_video.read()
    if not ret:
      break

    print(f"\rProcessing frame {frameNumber}", end="")
    frameNumber += 1

    frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    frame = torch.from_numpy(frame).to(device).float()

    distorted_frame = remap.torch_remap(distortion_map, frame)
    distorted_frame = helpers.centerCrop(distorted_frame, output_width, output_height)
    output_video.write_frame(distorted_frame)
  print()

  input_video.release()
  print("Saving video...")
  output_video.save_video()

if __name__ == "__main__":
  main()
