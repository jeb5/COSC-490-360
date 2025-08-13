import math
import os
from DataManager import DataManager
from FeatureDetector import FeatureManager
from helpers import ProcessContext
import progressbar as pb
import cv2 as cv
import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R
from scipy.sparse.linalg import svds
import scipy.sparse

import helpers


def rotation_chaining(dm: DataManager, feature_manager: FeatureManager, produce_debug=False):
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
      if produce_debug:
        dm.write_debug_frame(draw_matches(dm.get_frame(frame_number, undistort=True), matches, features))
      previous_features = features
      estimated_orientation_change, _ = estimate_orientation_change(matches, intrinsic_matrix)
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
  MAX_CACHE = 100
  n = dm.get_sequence_length()

  # if os.path.exists("temp/matches.txt"):
  #   print("Loading matches from temp/matches.txt")
  #   relative_rotations = []
  #   with open("temp/matches.txt", "r") as f:
  #     for line in f:
  #       i, j, rotation = line.strip().split(",", maxsplit=2)
  #       i, j = int(i), int(j)
  #       rotation = R.from_matrix(np.array(eval(rotation)))
  #       relative_rotations.append((i, j, rotation))
  # else:
  intrinsic_matrix, _, _ = dm.get_camera_info()
  back_sequence = helpers.get_sequence(window_size, window_size // 3, 3000) if quadratic else range(1, window_size + 1)
  loop_states = []
  for i in range(n):
    for j in (i - x for x in back_sequence):
      if j >= 0:
        loop_states.append((i, j))

  cached_features = {}
  feature_cache_queue = []

  prefix_widgets = [f"Sliding window matching {'(quadratic) ' if quadratic else ''}"]

  relative_rotations = []

  with ProcessContext(prefix_widgets=prefix_widgets, max_value=len(loop_states)) as bar:
    for i, j in bar(loop_states):
      for k in [i, j]:
        if k not in cached_features:
          cached_features[k] = feature_manager.detect_features(k)
          feature_cache_queue.append(k)
        if len(feature_cache_queue) > MAX_CACHE:
          del cached_features[feature_cache_queue.pop(0)]
      matches = feature_manager.get_matches(cached_features[i], cached_features[j])
      estimated_orientation_change, estimation_info = estimate_orientation_change(matches, intrinsic_matrix)
      if is_valid_estimation(estimation_info):
        relative_rotations.append((i, j, estimated_orientation_change))
    cached_features.clear()

    # print("Serializing matches to temp/matches.txt")
    # with open("temp/matches.txt", "w") as f:
    #   for i, j, rotation in relative_rotations:
    #     f.write(f"{i},{j},{rotation.as_matrix().tolist()}\n")

  estimated_orientations = solve_absolute_orientations(relative_rotations, n, 0)
  correction_matrix = dm.get_inertial(0) * estimated_orientations[0].inv()
  estimated_orientations = [correction_matrix * rot for rot in estimated_orientations]
  return estimated_orientations


def overlapping_windows(dm, window_size, feature_manager):
  # Implement overlapping windows logic here
  MAX_CACHE = 100
  intrinsic_matrix, _, _ = dm.get_camera_info()

  assert window_size % 2 == 0, "Window size must be even for overlapping windows."
  half_window = window_size // 2
  estimated_orientations = []
  num_blocks = math.ceil((2.0 * dm.get_sequence_length()) / window_size) - 1
  cached_relative_orientations = {}
  cached_features = {}
  feature_cache_queue = []
  prefix_widgets = ["Overlapping windows matching | Block ", pb.Counter(format="%(value)d"), "/", pb.FormatLabel("%(max_value)d")]
  with ProcessContext(prefix_widgets=prefix_widgets, max_value=num_blocks) as bar:
    for block_num in bar(range(num_blocks)):
      start_frame = block_num * half_window
      block_length = min(window_size, dm.get_sequence_length() - start_frame)
      relative_rotations_list = []
      for i in range(0, block_length):
        for j in range(0, i):
          if (start_frame + i, start_frame + j) not in cached_relative_orientations:
            if (start_frame + i) not in cached_features:
              cached_features[start_frame + i] = feature_manager.detect_features(start_frame + i)
              feature_cache_queue.append(start_frame + i)
            if (start_frame + j) not in cached_features:
              cached_features[start_frame + j] = feature_manager.detect_features(start_frame + j)
              feature_cache_queue.append(start_frame + j)
            if len(feature_cache_queue) > MAX_CACHE:
              del cached_features[feature_cache_queue.pop(0)]
            matches = feature_manager.get_matches(cached_features[start_frame + i], cached_features[start_frame + j])
            estimated_orientation_change, estimation_info = estimate_orientation_change(matches, intrinsic_matrix)
            if is_valid_estimation(estimation_info):
              cached_relative_orientations[(start_frame + i, start_frame + j)] = estimated_orientation_change
          estimated_orientation_change = cached_relative_orientations[(start_frame + i, start_frame + j)]
          relative_rotations_list.append((i, j, estimated_orientation_change))
      block_estimated_orientations = solve_absolute_orientations(relative_rotations_list, block_length, 0)
      block_correction_matrix = None
      if block_num == 0:
        block_correction_matrix = dm.get_inertial(0) * block_estimated_orientations[0].inv()
      else:
        block_correction_matrix = estimated_orientations[-half_window] * block_estimated_orientations[0].inv()
      block_estimated_orientations = [block_correction_matrix * rot for rot in block_estimated_orientations]
      if block_num == 0:
        estimated_orientations.extend(block_estimated_orientations)
      else:
        estimated_orientations.extend(block_estimated_orientations[half_window:])
  return estimated_orientations


def estimate_orientation_change(point_matches, intrinsic_matrix):
  points1, points2 = point_matches
  if len(points1) < 4:
    return None, None
  H, mask = cv.findHomography(points1, points2, cv.USAC_MAGSAC, 0.25)
  if H is None:
    return None, None

  inlier_points1 = points1[mask.ravel() == 1]
  inlier_points2 = points2[mask.ravel() == 1]

  if len(inlier_points1) < 4:
    return None, None

  H, _ = cv.findHomography(inlier_points1, inlier_points2, 0)  # Least squares refinement

  extracted_rotation = np.linalg.inv(intrinsic_matrix) @ H @ intrinsic_matrix
  orthonormalized_rotation = orthonormalize(extracted_rotation)

  if np.linalg.det(orthonormalized_rotation) < 0:
    # TODO: Just flip the sign of one of the columns?
    return None, None

  rotation = convert_coordinate_system(orthonormalized_rotation)
  rotation = R.from_matrix(rotation)

  estimation_info = {
    "inliers": len(inlier_points1),
    "angle_change": rotation.magnitude() * (180 / np.pi),
  }

  return rotation, estimation_info


def convert_coordinate_system(rotation):
  return np.array(
    [
      [rotation[0, 0], rotation[2, 0], -rotation[1, 0]],
      [rotation[0, 2], rotation[2, 2], -rotation[1, 2]],
      [-rotation[0, 1], -rotation[2, 1], rotation[1, 1]],
    ]
  )

def draw_matches(frame, matches, features):
  pts1, pts2 = matches
  kpts2, _ = features

  output = cv.drawKeypoints(
    frame,
    kpts2,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
  )
  for i in range(len(pts1)):
    pt1 = tuple(map(int, pts1[i]))
    pt2 = tuple(map(int, pts2[i]))
    cv.line(output, pt1, pt2, (0, 0, 255), 1)
  return output

def orthonormalize(rotation):
  U, _, Vt = np.linalg.svd(rotation)
  D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
  R = U @ D @ Vt
  return R


def is_valid_estimation(estimation_info):
  if estimation_info is None:
    return False
  if estimation_info["inliers"] < 10:
    return False
  if estimation_info["angle_change"] > 70:
    return False
  return True


def solve_absolute_orientations(observed_relative_rotations, n, critical_group_index):
  G = nx.Graph()
  for i, j, observed_rotation in observed_relative_rotations:
    G.add_edge(i, j)

  critical_group = list(nx.node_connected_component(G, critical_group_index))
  if len(critical_group) == 0:
    return [None] * n
  connected_matches = [
    (i, j, observed_rotation)
    for i, j, observed_rotation in observed_relative_rotations
    if i in critical_group and j in critical_group
  ]
  mask = np.zeros((n,), dtype=bool)
  mask[critical_group] = True

  m = len(connected_matches)
  A = np.zeros((m * 3, n * 3))
  for idx, (i, j, observed_rotation) in enumerate(connected_matches):
    A[idx * 3 : idx * 3 + 3, i * 3 : i * 3 + 3] = -observed_rotation.inv().as_matrix()
    A[idx * 3 : idx * 3 + 3, j * 3 : j * 3 + 3] = np.eye(3)

  A_sp = scipy.sparse.csr_matrix(A)
  _, _, Vt = svds(A_sp, k=3, which="SM")
  solution_matricies = np.stack([Vt[-1], Vt[-2], Vt[-3]], axis=1)
  solution_matricies = np.vsplit(solution_matricies, n)

  estimated_orientations = [
    R.from_matrix(orthonormalize(rotation)).inv() if mask[idx] else None for idx, rotation in enumerate(solution_matricies)
  ]
  return estimated_orientations