import cv2 as cv
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime

import helpers
from video_writer import VideoWriter


class DataManager:
  def __init__(self, directory, input_frame_interval=1, input_frame_scale=1.0):
    self.directory = directory
    self.input_frame_interval = input_frame_interval
    self.input_frame_scale = input_frame_scale
    input_video_path = os.path.join(directory, "raw.mp4")
    self.input_video = cv.VideoCapture(input_video_path, cv.CAP_FFMPEG)
    if not self.input_video.isOpened():
      raise Exception(f"Could not open video file: {input_video_path}")
    camera_info_path = os.path.join(directory, "camera_information.json")
    if not os.path.exists(camera_info_path):
      raise FileNotFoundError(f"Camera info file not found: {camera_info_path}")
    with open(camera_info_path, "r") as f:
      camera_info = json.load(f)
      self.intrinsic_matrix = np.array(camera_info["intrinsic_matrix"], dtype=np.float32)
      self.distortion_coefficients = np.array(camera_info["distortion_coefficients"], dtype=np.float32)
    inertial_path = os.path.join(directory, "inertials.csv")
    if not os.path.exists(inertial_path):
      raise FileNotFoundError(f"Inertial data file not found: {inertial_path}")
    self.inertial_rotations = [
      R.from_euler("ZXY", [xyz[2], xyz[0], xyz[1]], degrees=True) for frame, xyz in helpers.rotations_from_csv(inertial_path)
    ]
    self.num_frames = int(self.input_video.get(cv.CAP_PROP_FRAME_COUNT))
    if self.num_frames != len(self.inertial_rotations):
      raise ValueError(
        f"Number of inertial frames ({len(self.inertial_rotations)}) does not match video frames ({self.num_frames})"
      )

    self.input_size = (
      int(self.input_video.get(cv.CAP_PROP_FRAME_WIDTH)),
      int(self.input_video.get(cv.CAP_PROP_FRAME_HEIGHT)),
    )

    if len(self.distortion_coefficients) > 0:
      # self.undistorted_cam_mat = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
      #   self.intrinsic_matrix, self.distortion_coefficients, self.input_size, None, None, 1, self.input_size, 1
      # )
      self.m1, self.m2 = cv.fisheye.initUndistortRectifyMap(
        self.intrinsic_matrix, self.distortion_coefficients, None, self.intrinsic_matrix, self.input_size, cv.CV_32FC1
      )

    self.frame_rate = self.input_video.get(cv.CAP_PROP_FPS)

    now = datetime.now().strftime("%y%m%d%H%M%S")
    self.output_dir = os.path.join(directory, "outputs", f"output_{now}")
    os.makedirs(self.output_dir, exist_ok=True)
    self.equirectangular_video_path = os.path.join(self.output_dir, "360.mp4")
    self.debug_video_path = os.path.join(self.output_dir, "debug.mp4")
    self.orientations_path = os.path.join(self.output_dir, "orientations.csv")
    self.output_360_video = None
    self.output_debug_video = None

  def get_frame(self, frame_number, undistort=False):
    real_frame_number = frame_number * self.input_frame_interval
    self.input_video.set(cv.CAP_PROP_POS_FRAMES, real_frame_number)
    ret, image = self.input_video.read()
    if not ret:
      raise Exception(f"Failed to read frame {real_frame_number} from video.")
    if undistort:
      image = cv.remap(image, self.m1, self.m2, cv.INTER_LINEAR)
    if self.input_frame_scale != 1.0:
      image = cv.resize(image, None, fx=self.input_frame_scale, fy=self.input_frame_scale, interpolation=cv.INTER_LINEAR)
    return image

  def get_inertial(self, frame_number):
    return self.inertial_rotations[frame_number * self.input_frame_interval]

  def get_inertials(self):
    return [self.get_inertial(i) for i in range(self.get_sequence_length())]

  def get_sequence_length(self):
    return self.num_frames // self.input_frame_interval

  def write_360_frame(self, frame):
    # Frame is a tensor
    if self.output_360_video is None:
      frame_size = (frame.shape[1], frame.shape[0])
      self.output_360_video = VideoWriter(self.equirectangular_video_path, self.frame_rate, frame_size, spherical_metadata=True)
    self.output_360_video.write_frame(frame)

  def save_360_video(self):
    if self.output_360_video is not None:
      self.output_360_video.save_video()

  def write_debug_frame(self, frame):
    if self.output_debug_video is None:
      frame_size = (frame.shape[1], frame.shape[0])
      self.output_debug_video = VideoWriter(self.debug_video_path, self.frame_rate, frame_size)
    self.output_debug_video.write_frame(frame)

  def save_debug_video(self):
    if self.output_debug_video is not None:
      self.output_debug_video.save_video()

  def get_camera_info(self):
    return self.intrinsic_matrix, self.distortion_coefficients, self.input_size

  def write_orientations(self, orientations):
    with open(self.orientations_path, "w") as f:
      f.write("frame_number,pitch,roll,yaw\n")
      for frame_number, (pitch, roll, yaw) in enumerate(orientations):
        f.write(f"{frame_number},{pitch},{roll},{yaw}\n")