from DataManager import DataManager
from FeatureDetector import FeatureManager
from helpers import ProcessContext
import progressbar as pb
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_chaining(dm: DataManager, feature_manager: FeatureManager):
  estimated_orientations = []

  prefix_widgets = [
    "Rotation estimation with chaining | Frame ",
    pb.Counter(format="%(value)d"),
    "/",
    pb.FormatLabel("%(max_value)d"),
  ]
  with ProcessContext(prefix_widgets=prefix_widgets, max_value=dm.get_sequence_length()) as bar:
    estimated_orientations = [dm.get_inertial(0)]  # Start with the first inertial orientation
    previous_features = feature_manager.detect_features(0)
    intrinsic_matrix, _, _ = dm.get_camera_info()

    print()
    for frame_number in bar(range(1, dm.get_sequence_length())):
      features = feature_manager.detect_features(frame_number)
      matches = feature_manager.get_matches(previous_features, features)
      previous_features = features
      estimated_orientation_change = estimate_orientation_change(matches, intrinsic_matrix)
      if estimated_orientation_change is None:
        break
      estimated_orientation = estimated_orientations[-1] * estimated_orientation_change
      estimated_orientations.append(estimated_orientation)
      yaw, pitch, roll = estimated_orientation.as_euler("ZXY")
      td = 180 / np.pi
      message = f"Frame: {frame_number}, Pitch: {pitch * td:.2f}˚, Roll: {roll * td:.2f}˚, Yaw: {yaw * td:.2f}˚"
      print(f"\033[F\033[K{message}\033[E", end="")
  print()

  return estimated_orientations


def sliding_window(dm, window_size, feature_manager, quadratic=False):
  # Implement sliding window logic here
  pass


def overlapping_windows(dm, window_size, feature_manager):
  # Implement overlapping windows logic here
  pass


def estimate_orientation_change(point_matches, intrinsic_matrix):
  points1, points2 = point_matches
  if len(points1) < 4:
    return None
  H, mask = cv.findHomography(points1, points2, cv.USAC_MAGSAC, 0.25)
  if H is None:
    return None

  inlier_points1 = points1[mask.ravel() == 1]
  inlier_points2 = points2[mask.ravel() == 1]

  H, _ = cv.findHomography(inlier_points1, inlier_points2, 0)  # Least squares refinement

  extracted_rotation = np.linalg.inv(intrinsic_matrix) @ H @ intrinsic_matrix
  U, _, Vt = np.linalg.svd(extracted_rotation)
  orthonormalized_rotation = U @ Vt

  if np.linalg.det(orthonormalized_rotation) < 0:
    # TODO: Just flip the sign of one of the columns?
    return None

  # TODO: BUG: Filtering!
  # TODO: Debug video

  rotation = convert_coordinate_system(orthonormalized_rotation)
  return R.from_matrix(rotation)


def convert_coordinate_system(rotation):
  return np.array(
    [
      [rotation[0, 0], rotation[2, 0], -rotation[1, 0]],
      [rotation[0, 2], rotation[2, 2], -rotation[1, 2]],
      [-rotation[0, 1], -rotation[2, 1], rotation[1, 1]],
    ]
  )
