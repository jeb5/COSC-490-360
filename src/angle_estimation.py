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


  number_of_lines = -1  # Skip header
  with open(args.gyro_csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      number_of_lines += 1
  gyro = np.zeros((number_of_lines, 3)).astype(np.float32)
  pitch_yaw_roll = np.zeros((number_of_lines, 3)).astype(np.float32)
  acc = np.zeros((number_of_lines, 3)).astype(np.float32)
  frame_number = np.zeros((number_of_lines, 1)).astype(np.uint32)
  print("reading gyro data from CSV file...")
  with open(args.gyro_csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for i, row in enumerate(reader):
      frame_number[i] = [float(row[0])]
      # XYZ
      gyro[i] = [float(row[2]), float(row[3]), float(row[4])]
      # Pitch, Yaw, Roll
      pitch_yaw_roll[i] = [float(row[5]), float(row[6]), float(row[7])]
      # XYZ
      acc[i] = [float(row[8]), float(row[9]), float(row[10])]

  print("gyro data read from CSV file")
  input_video = cv.VideoCapture(args.video_path)
  if not input_video.isOpened():
    print("Error opening video file")
    return
  input_size = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  last_frame = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))
  input_framerate = int(input_video.get(cv.CAP_PROP_FPS))
  if args.output_video is not None:
    output_video = VideoWriter(args.output_video, input_framerate, input_size)
  i = 0
  (start_frame, end_frame) = (args.start_frame, args.end_frame if args.end_frame >= 0 else last_frame)

  cam_matrix, cam_distortion = helpers.GOPRO_CAMERA
  newMat = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(cam_matrix, cam_distortion, input_size, None, None, 1, input_size, 1)
  m1, m2 = cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, newMat, input_size, cv.CV_32FC1)
  m1, m2 = torch.from_numpy(m1), torch.from_numpy(m2)
  undistortMap = torch.stack((m1, m2), dim=-1)
  undistortMap = remap.absoluteToRelative(undistortMap, input_size)

  def cleanup():
    input_video.release()
    if args.output_video is not None:
      output_video.save_video()
    print("Done.")

  def interuppt_handler(signum, frame):
    cleanup()
    sys.exit(0)

  signal.signal(signal.SIGINT, interuppt_handler)

  initial_rotation = None
  visual_rotation = R.from_euler('zyx', np.array([0, 0, 0]))

  last_frame = None
  angle_vector_history = np.empty((0, 3))
  new_fangled_rotation = R.from_euler('ZYX', np.array([0, 0, 0])).as_matrix()
  with open(args.output_angle_path, 'w') as csvfile:
    for frame in range(start_frame, end_frame, 1):
      indices = np.where(frame_number == frame)[0]
      pitch_yaw_rolls = pitch_yaw_roll[indices]
      # average_pitch_yaw_roll = np.mean(pitch_yaw_rolls, axis=0)
      pitch_yaw_roll_frame = pitch_yaw_rolls[0]

      pitch = pitch_yaw_roll_frame[0]
      yaw = pitch_yaw_roll_frame[1]
      roll = pitch_yaw_roll_frame[2]
      yaw, roll = roll, yaw  # These seem to be swapped in the CSV file
      # pitch = pitch - 90
      roll = -roll
      # Get video frame
      input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
      ret, image = input_video.read()

      image_undistorted = remap.torch_remap(undistortMap, torch.from_numpy(image).float()).byte().numpy()
      angle_difference, matchImage = np.zeros((3, 1)), image_undistorted
      if last_frame is None:
        initial_rotation = R.from_euler('YXZ', np.array([yaw, pitch, roll]))
        new_fangled_rotation = R.from_euler('XYZ', np.array([yaw, pitch, roll])).as_matrix()
      else:
        angle_difference, matchImage, real_rot_mat = get_angle_difference(last_frame, image_undistorted, newMat)
        new_fangled_rotation = new_fangled_rotation @ real_rot_mat
        angle_vector = (new_fangled_rotation @ np.array([0, 0, 1]).T).T
        print("angle_vector", angle_vector)
        angle_vector_history = np.append(angle_vector_history, [angle_vector], axis=0)
        visual_rotation = visual_rotation * angle_difference
        vector_plot = generate_vector_history_plot(angle_vector_history)
        y_offset, y_end = int(matchImage.shape[0] - vector_plot.shape[0]), int(matchImage.shape[0])
        x_offset, x_end = int(matchImage.shape[1] - vector_plot.shape[1]), int(matchImage.shape[1])
        matchImage[y_offset:y_end, x_offset:x_end] = vector_plot

      visual_rotation_print = visual_rotation.as_euler('YXZ', degrees=True)
      total_rotation = initial_rotation * visual_rotation
      total_rotation_print = total_rotation.as_euler('YXZ', degrees=True)

      new_fangled_rotation_print = R.from_matrix(new_fangled_rotation).as_euler('YXZ', degrees=True)
      (pitch, yaw, roll) = new_fangled_rotation_print
      last_frame = image_undistorted

      print(f"Writing frame {i + 1}/{end_frame - start_frame}")
      if not ret:
        print("Error reading video frame")
        break
      if args.output_video is not None:
        if args.debug:
          angleText = "\
Visual Pitch: {:.1f}\n\
Visual Yaw: {:.1f}\n\
Visual Roll: {:.1f}\n\
Pitch: {:.1f}\n\
Yaw: {:.1f}\n\
Roll: {:.1f}".format(
            visual_rotation_print[1], visual_rotation_print[0], visual_rotation_print[2], pitch, yaw, roll)
          image_debug = add_text_to_image(matchImage, angleText)
          output_video.write_frame_opencv(image_debug)
        else:
          output_video.write_frame_opencv(image)

      toRad = np.pi / 180
      csvfile.write("{},{},{},{}\n".format(i, pitch * toRad, yaw * toRad, roll * toRad))
      i += 1
  cleanup()


def add_text_to_image(image, text):
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  pil_image = PIL.Image.fromarray(image)
  draw = PIL.ImageDraw.Draw(pil_image)
  font = PIL.ImageFont.truetype("src/assets/roboto.ttf", 30)
  draw.text((10, 10), text, fill=(255, 255, 255), font=font)
  image_cv = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
  return image_cv


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

  matchImage = frame2.copy()
  for i in range(len(good_points_1)):
    if mask[i]:
      pt1 = tuple(map(int, good_points_1[i]))
      pt2 = tuple(map(int, good_points_2[i]))
      cv.line(matchImage, pt1, pt2, (0, 0, 255), 1)

  ret, rotations, translations, normals = cv.decomposeHomographyMat(H, cameraMatrix)
  real_rot_mat = rotations[0]
  rotation = R.from_matrix(rotations[0])
  return (rotation, matchImage, real_rot_mat)


def generate_vector_history_plot(vector_history):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  xs, ys, zs = vector_history[:, 0], vector_history[:, 1], vector_history[:, 2]
  ax.scatter(xs, ys, zs, c='red', marker='o')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.set_zlim(-1, 1)
  # Draw last vector as an arrow
  ax.quiver(0, 0, 0, xs[-1], ys[-1], zs[-1], length=1, normalize=True, color='red')
  # return plot as image
  # fig.show()
  fig.canvas.draw()
  image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
  w, h = fig.canvas.get_width_height()
  image = image.reshape(h * 2, w * 2, 4)
  image = image[:, :, 1:4]
  plt.close(fig)
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
