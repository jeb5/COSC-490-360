import cv2 as cv
import remap
import helpers
import torch
from VideoWriter import VideoWriter
import argparse


def main(args):
  input_video = cv.VideoCapture(args.video_path)
  if not input_video.isOpened():
    print("Error: Could not open video.")
    return
  cam_mat, cam_distortion = helpers.BLENDER_CAMERA_WITH_FISHEYE
  cam_mat[0][2] = cam_mat[0][2] * args.overscan
  cam_mat[1][2] = cam_mat[1][2] * args.overscan

  in_w, in_h = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  frame_count = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))
  out_w, out_h = int(in_w / args.overscan), int(in_h / args.overscan)

  output_video = VideoWriter(args.output_path, 20, (out_w, out_h), mbps=10)

  device = helpers.get_device()

  distortion_map = remap.getFisheyeDistortionMap((in_w, in_h), cam_mat, cam_distortion).to(device)

  frameNumber = 0
  while True:
    ret, frame = input_video.read()
    if not ret:
      break

    print(f"\rProcessing frame {frameNumber}/{frame_count}", end="")
    frameNumber += 1

    frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    frame = torch.from_numpy(frame).to(device).float()

    distorted_frame = remap.torch_remap(distortion_map, frame)
    distorted_frame = helpers.centerCrop(distorted_frame, out_w, out_h)
    output_video.write_frame(distorted_frame)
  print()

  input_video.release()
  print("Saving video...")
  output_video.save_video()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Remap a video using fisheye distortion.")
  parser.add_argument("video_path", type=str, help="Path to the input video file.")
  parser.add_argument("output_path", type=str, help="Path to the output video file.")
  parser.add_argument("--overscan", type=float, default=1.0, help="Percentage of overscan in input video. 1.0 = normal. 1.5 = 50%% extra overscan.")
  args = parser.parse_args()
  main(args)
