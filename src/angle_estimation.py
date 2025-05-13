import csv
import math
import numpy as np
import cv2 as cv
import PIL.ImageDraw
import PIL.ImageFont
import PIL.Image
import matplotlib.pyplot as plt
import argparse
import helpers
import torch
import remap
import sys
import signal
from video_writer import VideoWriter
from scipy.spatial.transform import Rotation as R


def main(args):

  input_video = cv.VideoCapture(args.video_path)
  if not input_video.isOpened():
    print("Error opening video file")
    return

  input_size = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  input_framerate = int(input_video.get(cv.CAP_PROP_FPS))
  (start_frame, end_frame) = (args.start_frame, args.end_frame if args.end_frame >= 0 else int(input_video.get(cv.CAP_PROP_FRAME_COUNT)))

  cam_matrix, cam_distortion = helpers.GOPRO_CAMERA
  sensible_undistorted_cam_mat = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
    cam_matrix, cam_distortion, input_size, None, None, 1, input_size, 1)
  m1, m2 = cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, sensible_undistorted_cam_mat, input_size, cv.CV_32FC1)

  if args.output_video is not None:
    output_video = VideoWriter(args.output_video, input_framerate, input_size)

  def cleanup():
    input_video.release()
    if args.output_video is not None:
      output_video.save_video()
    print("Done.")

  def interupt_handler(signum, frame):
    cleanup()
    sys.exit(0)
  signal.signal(signal.SIGINT, interupt_handler)

  visual_rotations = []
  inertial_rotations = []
  previous_frame = None
  with open(args.output_angle_path, 'w') as csvfile:
    for (frame, yaw_pitch_roll) in inertials_from_csv(args.gyro_csv_path, start_frame, end_frame):

      input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
      ret, image = input_video.read()
      if not ret:
        print("Error reading video frame")
        break
      image_undistorted = cv.remap(image, m1, m2, cv.INTER_LINEAR)

      yaw_pitch_roll = np.array(yaw_pitch_roll)
      yaw_pitch_roll += np.array([0, -90, 0])
      yaw_pitch_roll *= np.array([1, -1, 1])

      inertial_rotations.append(R.from_euler('YXZ', yaw_pitch_roll, degrees=True).as_matrix())
      match_image = None
      if frame == start_frame:
        visual_rotations.append(inertial_rotations[-1])
        match_image = image_undistorted.copy()
      else:
        visual_rotation_change, match_image = get_angle_difference(previous_frame, image_undistorted, sensible_undistorted_cam_mat)
        visual_rotations.append(visual_rotations[-1] @ visual_rotation_change)

      vector_plot = generate_rotation_histories_plot([{"name": "Visual", "colour": "#42a7f5", "data": visual_rotations}, {
                                                     "name": "Inertial", "colour": "#f07d0a", "data": inertial_rotations}])
      (x, y) = match_image.shape[1] - vector_plot.shape[1], match_image.shape[0] - vector_plot.shape[0]
      match_image = helpers.paste_cv(match_image, vector_plot, x, y)

      print(f"Writing frame {frame - start_frame + 1}/{end_frame - start_frame}")

      if args.output_video is not None:
        if args.debug:
          angleText = f"Inertial Yaw: {yaw_pitch_roll[0]:.2f}, Pitch: {yaw_pitch_roll[1]:.2f}, Roll: {yaw_pitch_roll[2]:.2f}"
          visual_rot_YXZ = R.from_matrix(visual_rotations[-1]).as_euler('YXZ', degrees=True)
          angleText += f"\nVisual   Yaw: {visual_rot_YXZ[0]:.2f}, Pitch: {visual_rot_YXZ[1]:.2f}, Roll: {visual_rot_YXZ[2]:.2f}"
          image_debug = add_text_to_image(match_image, angleText)
          output_video.write_frame_opencv(image_debug)
        else:
          output_video.write_frame_opencv(image)
      previous_frame = image_undistorted.copy()

      # toRad = np.pi / 180
      # csvfile.write("{},{},{},{}\n".format(i, pitch * toRad, yaw * toRad, roll * toRad))
  cleanup()


def add_text_to_image(image, text):
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  pil_image = PIL.Image.fromarray(image)
  draw = PIL.ImageDraw.Draw(pil_image)
  font = PIL.ImageFont.truetype("src/assets/roboto.ttf", 30)
  draw.text((10, 10), text, fill=(255, 255, 255), font=font)
  image_cv = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
  return image_cv


def inertials_from_csv(csv_path, first_frame, last_frame):
  current_frame = first_frame
  frame_with_yaw_pitch_roll = None
  with open(csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    # skip header
    next(reader)
    for i, row in enumerate(reader):
      frame_number = int(row[0])
      if frame_number < first_frame:
        continue
      if frame_number > last_frame:
        yield frame_with_yaw_pitch_roll
        break
      if frame_number > current_frame:
        current_frame = frame_number
        yield frame_with_yaw_pitch_roll
      frame_with_yaw_pitch_roll = (current_frame, (float(row[7]), float(row[5]), float(row[6])))

def get_angle_difference(frame1, frame2, cameraMatrix):
  sift = cv.SIFT_create()

  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(frame1, None)
  kp2, des2 = sift.detectAndCompute(frame2, None)

  # FLANN parameters
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)

  flann = cv.FlannBasedMatcher(index_params, search_params)

  matches = flann.knnMatch(des1, des2, k=2)

  # # Need to draw only good matches, so create a mask
  matchesMask = [[0, 0] for i in range(len(matches))]

  good_points_1 = []
  good_points_2 = []
  for i, (m, n) in enumerate(matches):
      if m.distance < 0.7 * n.distance:
          matchesMask[i] = [1, 0]
          good_points_1.append(kp1[m.queryIdx].pt)
          good_points_2.append(kp2[m.trainIdx].pt)
  good_points_1 = np.array(good_points_1)
  good_points_2 = np.array(good_points_2)

  if len(good_points_1) < 4:
    print("Not enough good points to compute homography")
    return np.zeros((3, 1)), frame2
  H, mask = cv.findHomography(good_points_1, good_points_2, cv.RANSAC, 5.0)

  match_image = frame2.copy()
  for i in range(len(good_points_1)):
    if mask[i]:
      pt1 = tuple(map(int, good_points_1[i]))
      pt2 = tuple(map(int, good_points_2[i]))
      cv.line(match_image, pt1, pt2, (0, 0, 255), 1)

  ret, rotations, translations, normals = cv.decomposeHomographyMat(H, cameraMatrix)
  real_rot_mat = rotations[0]
  return real_rot_mat, match_image
  # rotation = R.from_matrix(rotations[0])
  # return (rotation, matchImage, real_rot_mat)


def generate_rotation_histories_plot(rotation_histories):
  fig = plt.figure(figsize=(4, 4))  # Make the figure smaller
  ax = fig.add_subplot(111, projection="3d")
  for rotation_history in rotation_histories:
    name = rotation_history["name"]
    colour = rotation_history["colour"]
    rotations = rotation_history["data"]
    vectors = [rotation @ np.array([0, 0, 1]) for rotation in rotations]
    vector_history = np.array(vectors)
    xs, ys, zs = vector_history[:, 0], vector_history[:, 1], vector_history[:, 2]
    ax.quiver(0, 0, 0, xs[-1], ys[-1], zs[-1], length=1, normalize=True, color=colour, arrow_length_ratio=0.2)
    ax.scatter(xs, ys, zs, c=colour, marker='.', label=name)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.set_zlim(-1, 1)
  ax.legend()

  fig.canvas.draw()
  w, h = fig.canvas.get_width_height(physical=True)
  plt.close(fig)
  image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
  image = image.reshape(h, w, 4)
  image = image[:, :, 1:4]
  image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  return image

# ZYX = YAW, PITCH, ROLL
# Yaw around Z, Pitch around Y, Roll around X


def Rz(alpha):
  return torch.tensor([[math.cos(alpha), -math.sin(alpha), 0],
                      [math.sin(alpha), math.cos(alpha), 0],
                      [0, 0, 1]])


def Ry(beta):
  return torch.tensor([[math.cos(beta), 0, math.sin(beta)],
                      [0, 1, 0],
                      [-math.sin(beta), 0, math.cos(beta)]])


def Rx(gamma):
  return torch.tensor([[1, 0, 0],
                      [0, math.cos(gamma), -math.sin(gamma)],
                      [0, math.sin(gamma), math.cos(gamma)]])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Estimate angles from gyro data and video.")
  parser.add_argument("video_path", type=str, help="Path to the input video file.")
  parser.add_argument("gyro_csv_path", type=str, help="Path to the gyro CSV file.")
  parser.add_argument("output_angle_path", type=str, help="Path to the output angle CSV file.")
  parser.add_argument("--output_video", type=str, help="Path to the output video file.")
  parser.add_argument("--start_frame", type=int, default=0, help="Start frame for processing.")
  parser.add_argument("--end_frame", type=int, default=-1, help="End frame for processing.")
  parser.add_argument("--debug", action='store_true', help="Enable debug mode to ouput angles, matches and undistorted image.")
  args = parser.parse_args()
  main(args)
