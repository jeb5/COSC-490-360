import math

import torch
from DataManager import DataManager
from FeatureManager import FeatureManager
from ObservationManager import ObservationManager
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
  intrinsic_matrix, _, _ = dm.get_camera_info()

  def orientation_estimation_func(matches):
    return estimate_orientation_change(matches, intrinsic_matrix)

  observation_manager = ObservationManager(feature_manager, is_valid_estimation, orientation_estimation_func)

  back_sequence = helpers.get_sequence(window_size, window_size // 3, 3000) if quadratic else range(1, window_size + 1)
  frame_pairs = []
  for j in range(dm.get_sequence_length()):
    for i in (j - x for x in back_sequence):
      if i >= 0:
        frame_pairs.append((i, j))

  prefix_widgets = [f"Sliding window matching {'(quadratic) ' if quadratic else ''}"]
  relative_rotations = []

  with ProcessContext(prefix_widgets=prefix_widgets, max_value=len(frame_pairs)) as bar:
    relation_image = np.zeros((dm.get_sequence_length(), dm.get_sequence_length(), 3), dtype=np.float32)

    for observation in bar(observation_manager.get_observations(frame_pairs)):
      relative_rotations.append(observation)
      i, j, rotation = observation
      inertial_ground_truth = dm.get_inertial(i).inv() * dm.get_inertial(j)
      difference = (rotation.inv() * inertial_ground_truth).magnitude() * (180 / np.pi)
      relation_image[i, j, 0] = min(difference * 1.2, 1.0)
      relation_image[i, j, 1] = 1

      if difference > 0.3:
        print(f"Significant difference found between frames {i} and {j}: {difference:.2f}˚")
        print(f"Inertial i: {dm.get_inertial(i).as_euler('ZXY', degrees=True)}")
        print(f"Inertial j: {dm.get_inertial(j).as_euler('ZXY', degrees=True)}")
        print(f"Ground truth: {inertial_ground_truth.as_euler('ZXY', degrees=True)}")
        print(f"Estimate: {rotation.as_euler('ZXY', degrees=True)}")
        # fi = feature_manager.detect_features(i)
        # fj = feature_manager.detect_features(j)
        # matches = feature_manager.get_matches(fi, fj)
        # debug_pic = draw_matches(dm.get_frame(i, undistort=True), matches, fi)
        # cv.imshow("Debug Matches", debug_pic)
        # cv.waitKey(0)
    # save relation_image
    cv.imwrite("temp/relation_image.png", (relation_image * 255).astype(np.uint8))

  estimated_orientations = solve_absolute_orientations(relative_rotations, dm.get_sequence_length(), 0)

  correction_matrix = dm.get_inertial(0) * estimated_orientations[0].inv()
  estimated_orientations = [correction_matrix * rot if rot is not None else rot for rot in estimated_orientations]
  return estimated_orientations


def overlapping_windows(dm, window_size, feature_manager):
  assert window_size % 2 == 0, "Window size must be even for overlapping windows."

  intrinsic_matrix, _, _ = dm.get_camera_info()

  def orientation_estimation_func(matches):
    return estimate_orientation_change(matches, intrinsic_matrix)

  observation_manager = ObservationManager(feature_manager, is_valid_estimation, orientation_estimation_func)

  half_window = window_size // 2
  s = window_size // 4
  t = half_window + s
  estimated_orientations = []
  last_block = []
  num_blocks = math.ceil((2.0 * dm.get_sequence_length()) / window_size) - 1
  prefix_widgets = ["Overlapping windows matching | Block ", pb.Counter(format="%(value)d"), "/", pb.FormatLabel("%(max_value)d")]
  with ProcessContext(prefix_widgets=prefix_widgets, max_value=num_blocks) as bar:
    for block_num in bar(range(num_blocks)):
      start_frame = block_num * half_window
      block_length = min(window_size, dm.get_sequence_length() - start_frame)
      relative_rotations = []
      for i, j, rotation in observation_manager.get_observations_in_window(start_frame, start_frame + block_length):
        relative_rotations.append((i - start_frame, j - start_frame, rotation.inv()))

        inertial_ground_truth = dm.get_inertial(i).inv() * dm.get_inertial(j)
        # relative_rotations.append((i - start_frame, j - start_frame, inertial_ground_truth))
        difference = (rotation.inv() * inertial_ground_truth).magnitude() * (180 / np.pi)

        # print(f"{i} -> {j}: {difference:.2f}˚")

        if difference > 1:
          print(f"Significant difference found between frames {i} and {j}: {difference:.2f}˚")
        #   print(f"Inertial i: {dm.get_inertial(i).as_euler('ZXY', degrees=True)}")
        #   print(f"Inertial j: {dm.get_inertial(j).as_euler('ZXY', degrees=True)}")
        #   print(f"Ground truth: {inertial_ground_truth.as_euler('ZXY', degrees=True)}")
        #   print(f"Estimate: {rotation.as_euler('ZXY', degrees=True)}")

      block_estimated_orientations = solve_absolute_orientations(relative_rotations, block_length, 0)
      block_correction_matrix = None
      if block_num == 0:
        block_correction_matrix = dm.get_inertial(0) * block_estimated_orientations[0].inv()
      else:
        block_correction_matrix = last_block[t] * block_estimated_orientations[s].inv()
        # block_correction_matrix = block_estimated_orientations[0].inv()
      block_estimated_orientations = [block_correction_matrix * rot for rot in block_estimated_orientations]
      if block_num == 0:
        estimated_orientations.extend(block_estimated_orientations)
      else:
        estimated_orientations.extend(block_estimated_orientations[half_window:])
      last_block = block_estimated_orientations
      # print(f"Block {block_num}: from {start_frame} to {start_frame + block_length}")
      # estimated_orientations.extend(block_estimated_orientations[:half_window])
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
  orthonormalized_rotation = project_to_so3(extracted_rotation)

  rotation = convert_coordinate_system(orthonormalized_rotation)
  rotation = R.from_matrix(rotation)

  estimation_info = {
    "inliers": len(inlier_points1),
    "angle_change": rotation.magnitude() * (180 / np.pi),
    "inliers_dice": (2 * len(inlier_points1)) / (len(points1) + len(points2)),
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


def is_valid_estimation(estimation_info):
  if estimation_info is None:
    return False
  if estimation_info["inliers"] < 20:
    return False
  if estimation_info["angle_change"] > 40:
    return False
  # if estimation_info["inliers_dice"] < 0.05:
  #   return False
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
    A[idx * 3 : idx * 3 + 3, i * 3 : i * 3 + 3] = -observed_rotation.as_matrix()
    A[idx * 3 : idx * 3 + 3, j * 3 : j * 3 + 3] = np.eye(3)

  A_sp = scipy.sparse.csr_matrix(A)
  _, _, Vt = svds(A_sp, k=3, which="SM")

  rotations = np.stack(Vt, axis=1).reshape(n, 3, 3)

  rotations = [orthonormalize(rot).T if rot is not None else None for rot in rotations]
  correction = rotations[0].T  # *Need* to correct matrices to fix det=-1
  rotations = [R.from_matrix(correction @ rot) if rot is not None else None for rot in rotations]
  return rotations


def project_to_so3(matrix):
  U, _, Vt = np.linalg.svd(matrix)
  D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
  R = U @ D @ Vt
  return R


def orthonormalize(matrix):
  U, _, Vt = np.linalg.svd(matrix)
  R = U @ Vt
  return R