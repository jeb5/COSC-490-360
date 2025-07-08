import numpy as np
import cv2 as cv
import argparse
import torch
from angle_estimation_helpers import (
  cache_features,
  entire_transformation,
  generate_rotation_histories_plot,
  get_features,
  get_homography_overlap_percent,
  load_features,
  overlay_homography,
  load_matches,
  cache_matches,
)
import helpers
import progressbar as pb
import sys
import signal
import networkx as nx
from video_writer import VideoWriter
from scipy.spatial.transform import Rotation as R
from scipy.sparse.linalg import eigsh, svds
import scipy.sparse


def main(args):
  input_video_path = helpers.get_file_path_pack_dir(args.directory, "video")
  input_inertial_path = helpers.get_file_path_pack_dir(args.directory, "inertial")
  output_visual_path = helpers.get_file_path_pack_dir(args.directory, "visual")

  cam_matrix, cam_distortion = helpers.load_camera_info(helpers.get_file_path_pack_dir(args.directory, "camera_info"))

  input_video = cv.VideoCapture(input_video_path)
  if not input_video.isOpened():
    print("Error opening video file")
    return

  input_size = (
    int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT)),
  )
  input_framerate = int(input_video.get(cv.CAP_PROP_FPS))
  start_frame, end_frame = (0, int(input_video.get(cv.CAP_PROP_FRAME_COUNT) - 1))

  new_cam_mat = (
    cv.fisheye.estimateNewCameraMatrixForUndistortRectify(cam_matrix, cam_distortion, input_size, None, None, 1, input_size, 1)
    if len(cam_distortion) > 0
    else cam_matrix
  )
  m1, m2 = (
    cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, new_cam_mat, input_size, cv.CV_32FC1)
    if len(cam_distortion) > 0
    else (None, None)
  )

  output_debug_video_path = helpers.get_file_path_pack_dir(args.directory, "debug_video") if args.output_debug_video else None
  output_debug_video = VideoWriter(output_debug_video_path, input_framerate, input_size) if args.output_debug_video else None
  # IF output_debug_video was disabled, we *could* save time by not undistorting the image, only the feature locations

  def cleanup():
    input_video.release()
    if output_debug_video is not None:
      if output_debug_video.did_write():
        output_debug_video.save_video()
        print(f"Debug video saved to {output_debug_video_path}")
    print("Done.")

  def interupt_handler(signum, frame):
    cleanup()
    sys.exit(0)

  signal.signal(signal.SIGINT, interupt_handler)
  try:
    inertial_rotations = [
      R.from_euler("ZXY", [xyz[2], xyz[0], xyz[1]], degrees=True).as_matrix()
      for frame, xyz in helpers.rotations_from_csv(input_inertial_path)
    ]
    inertial_rotations = inertial_rotations[start_frame : end_frame + 1]

    visual_rotations = []
    if args.naive_chaining:
      visual_rotations = chain_rotations(
        inertial_rotations, new_cam_mat, input_video, start_frame, end_frame, output_debug_video, m1, m2, args
      )
    else:
      visual_rotations = solve_rotations(new_cam_mat, inertial_rotations, input_video, start_frame, end_frame, m1, m2, args)

    # Output visual rotation (in xyz order)
    with open(output_visual_path, "w") as csvfile:
      csvfile.write("frame,pitch,roll,yaw\n")
      for frame, visual_rotation in enumerate(visual_rotations):
        if visual_rotation is None:
          continue
        visual_zxy = R.from_matrix(visual_rotation).as_euler("ZXY", degrees=True)
        csvfile.write(
          "{},{},{},{}\n".format(
            frame,
            visual_zxy[1],
            visual_zxy[2],
            visual_zxy[0],
          )
        )

    generate_rotation_histories_plot(
      [
        {"name": "Visual", "colour": "#42a7f5", "data": visual_rotations},
        {"name": "Inertial", "colour": "#f07d0a", "data": inertial_rotations},
      ],
      interactive=True,
    )

    total_angle_difference, count = 0.0, 0.0
    for i in range(len(visual_rotations)):
      if visual_rotations[i] is None or inertial_rotations[i] is None:
        continue
      count += 1
      difference = np.linalg.inv(visual_rotations[i]) @ inertial_rotations[i]
      total_angle_difference += np.linalg.norm(R.from_matrix(difference).as_rotvec(degrees=True))
    print(f"Average angle difference: {total_angle_difference / len(visual_rotations):.2f} degrees")
  finally:
    cleanup()


def chain_rotations(inertial_rotations, new_cam_mat, input_video, start_frame, end_frame, output_debug_video, m1, m2, args):
  visual_rotations = []

  frame_features = get_video_features(args, input_video, start_frame, end_frame, m1, m2)

  bar = pb.ProgressBar(
    max_value=(end_frame - start_frame + 1),
    widgets=[
      "Writing frame ",
      pb.Counter(format="%(value)d"),
      "/",
      pb.FormatLabel("%(max_value)d"),
      " ",
      pb.GranularBar(),
      " ",
      pb.ETA(),
    ],
    redirect_stdout=True,
    redirect_stderr=True,
  )
  for i in bar(range(end_frame - start_frame + 1)):
    frame = start_frame + i
    input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
    ret, image = input_video.read()
    if not ret:
      raise Exception(f"Failed to read frame {frame} from video.")
    image_undistorted = cv.remap(image, m1, m2, cv.INTER_LINEAR) if m1 is not None else image

    visual_rotation_change, match_image = None, None
    if i == 0:
      visual_rotations.append(inertial_rotations[0])
      match_image = image_undistorted.copy()
    else:
      visual_rotation_change, match_image, _ = get_angle_difference(
        frame_features[i - 1], frame_features[i], new_cam_mat, image_undistorted
      )
      if visual_rotation_change is None:
        print(f"Homography estimation failed at frame {frame}")
        visual_rotations.append(None)
      elif visual_rotations[i - 1] is None:
        # Use inertial rotation to continue the chain
        visual_rotations.append(inertial_rotations[i - 1] @ visual_rotation_change)
      else:
        visual_rotations.append(visual_rotations[i - 1] @ visual_rotation_change)

    visual_zxy = R.from_matrix(visual_rotations[i]).as_euler("ZXY", degrees=True) if visual_rotations[i] is not None else None
    inertial_zxy = R.from_matrix(inertial_rotations[i]).as_euler("ZXY", degrees=True)

    if output_debug_video is not None:
      angleText = ""
      angleText += f"Frame {i + 1}/{len(inertial_rotations)}"
      angleText += f"\nInertial:\n{inertial_zxy[0]:6.2f}y {inertial_zxy[1]:6.2f}p {inertial_zxy[2]:6.2f}r"
      if visual_rotation_change is not None:
        angleText += f"\nVisual:\n{visual_zxy[0]:6.2f}y {visual_zxy[1]:6.2f}p {visual_zxy[2]:6.2f}r"
        visual_zxy_change = R.from_matrix(visual_rotation_change).as_euler("ZXY", degrees=True)
        angleText += f"\nChange:\n{visual_zxy_change[0]:6.2f}y {visual_zxy_change[1]:6.2f}p {visual_zxy_change[2]:6.2f}r"
      vector_plot = generate_rotation_histories_plot(
        [
          {"name": "Visual", "colour": "#42a7f5", "data": visual_rotations[: i + 1]},
          {"name": "Inertial", "colour": "#f07d0a", "data": inertial_rotations[: i + 1]},
        ],
        extra_text=angleText,
      )
      if match_image is None:
        match_image = image_undistorted.copy()
      (x, y) = (
        match_image.shape[1] - vector_plot.shape[1],
        match_image.shape[0] - vector_plot.shape[0],
      )
      match_image = helpers.paste_cv(match_image, vector_plot, x, y)

      output_debug_video.write_frame_opencv(match_image)
  return visual_rotations


def get_video_features(args, input_video, start_frame, end_frame, m1, m2):
  features_cache_path = helpers.get_file_path_pack_dir(args.directory, "features_cache")
  frame_features = load_features(features_cache_path) if args.use_features_cache else []
  if len(frame_features) == 0:
    bar = pb.ProgressBar(
      max_value=(end_frame - start_frame + 1),
      widgets=["Finding SIFT features: ", pb.Percentage(), " ", pb.GranularBar(), " ", pb.ETA()],
    )
    for i in bar(range(end_frame - start_frame + 1)):
      frame = start_frame + i
      input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
      ret, image = input_video.read()
      image_undistorted = image if m1 is None else cv.remap(image, m1, m2, cv.INTER_LINEAR)
      frame_features.append(get_features(image_undistorted))
    cache_features(frame_features, features_cache_path)
  return frame_features


def find_matches(sift_features, cameraMatrix, inertial_rotations, window_size=None):
  frame_relations_picture = np.zeros((len(sift_features) * 2, len(sift_features), 3), dtype=np.uint8)
  matches = []
  n = len(sift_features)
  loop_states = []
  back_sequence = helpers.get_sequence(50, 10, 3000)
  for i in range(n):
    # for j in range(0, i) if window_size is None else range(max(0, i - window_size), i):
    for j in (i - x for x in back_sequence):
      if j < 0:
        continue
      loop_states.append((i, j))

  bar = pb.ProgressBar(
    max_value=len(loop_states),
    widgets=["Finding matches: ", pb.Percentage(), " ", pb.GranularBar(), " ", pb.ETA()],
    redirect_stdout=True,
  ).start()

  for idx, (i, j) in enumerate(loop_states):
    match_rot, _, extra_info = get_angle_difference(sift_features[j], sift_features[i], cameraMatrix)

    # intertial_rotation_change = np.linalg.inv(inertial_rotations[j]) @ inertial_rotations[i]
    # intertial_overlap_percent = get_homography_overlap_percent(intertial_rotation_change, cameraMatrix)
    # frame_relations_picture[i * 2 + 1, j, 1] = intertial_overlap_percent * 255
    if match_rot is not None:
      matches.append((i, j, match_rot))
      frame_relations_picture[i * 2, j, 0] = 255
      frame_relations_picture[i * 2, j, 2] = (min(extra_info["inliers_dice"] * 4, 1)) * 255

      # difference = np.linalg.inv(match_rot) @ intertial_rotation_change
      # angle_diff = np.linalg.norm(R.from_matrix(difference).as_rotvec(degrees=True))
      # frame_relations_picture[i * 2 + 1, j, 1] = min(angle_diff / 0.5, 1) * 255
      # total_angle_diff += angle_diff
      # if angle_diff > 0.5:
      #   print("Large angle difference detected:")
      #   print(f"  Frame {i} to {j}: {angle_diff:.2f} degrees")
      #   print(f"  Inliers: {extra_info['inliers']}, Dice: {extra_info['inliers_dice']:.2f}")
      #   print(f"  Orthonormality: {extra_info['orthonormality']:.2f}")

    bar.update(idx)
  bar.finish()
  # print(f"Average angle difference: {total_angle_diff / len(matches):.2f} degrees")
  # print(f"Total matches found: {len(matches)}")
  cv.imwrite("temp/frame_relations_new.png", frame_relations_picture)

  return matches


def solve_rotations(cameraMatrix, inertial_rotations, input_video, start_frame, end_frame, m1, m2, args):
  # Take list of frame features (SIFT lists)
  # For each frame, get with ALL previous frames (n^2, or more precisely, n(n-1)/2)
  # For each match pair, estimate rotation using homography. If not enough points, skip.
  # We will make a note of any frames that had no matches. (OR frames with no connection to the first frame. Union-Find?)
  # We then construct a matrix A using only the frames that are inter-matched.
  # To solve Az = 0, we construct A^TA, and take the 3 eigenvectors corresponding to the 3 smallest eigenvalues. This is equivilent to taking the 3 right singular vectors corresponding to the 3 smallest singular values of A
  # Now given 3 values of z, we reconstruct the rotation matrices R^{i} for each frame i.
  # These will be rotation matrices in a common coordinate frame, however not the world coordinate frame.
  # If we know the world coordinate frame of the first frame, we can rotate all matricies to the world coordinate frame.
  matches_cache_path = helpers.get_file_path_pack_dir(args.directory, "matches_cache")
  matches = load_matches(matches_cache_path) if args.use_matches_cache else []
  if len(matches) == 0:
    features = get_video_features(args, input_video, start_frame, end_frame, m1, m2)
    matches = find_matches(features, cameraMatrix, inertial_rotations, 10)
    cache_matches(matches, matches_cache_path)

  G = nx.Graph()
  for i, j, match_rot in matches:
    G.add_edge(i, j)

  connected_to_first = list(nx.node_connected_component(G, 0))
  m = max(connected_to_first) + 1
  # matches in connected graph (edges for which i and j are in connected_to_first)
  connected_matches = [(i, j, match_rot) for i, j, match_rot in matches if i in connected_to_first and j in connected_to_first]

  mask = np.zeros((m,), dtype=bool)
  mask[connected_to_first] = True

  A = torch.zeros((len(connected_matches) * 3, m * 3), dtype=torch.float32)
  for idx, (i, j, match_rot) in enumerate(connected_matches):
    print(idx, i, j, match_rot.shape)
    print("Shape of A:", A.shape)
    A[idx * 3 : idx * 3 + 3, i * 3 : i * 3 + 3] = -torch.from_numpy(match_rot)
    A[idx * 3 : idx * 3 + 3, j * 3 : j * 3 + 3] = torch.eye(3, dtype=torch.float32)
  # U, S, Vt = torch.linalg.svd(A)
  # z1 = Vt[-1]
  # z2 = Vt[-2]
  # z3 = Vt[-3]

  A_sp = A.to(torch.float64).cpu().numpy()
  A_sp = scipy.sparse.csr_matrix(A_sp)
  U, S, Vt = svds(A_sp, k=3, which="SM")
  z1 = torch.from_numpy(Vt[-1]).float()
  z2 = torch.from_numpy(Vt[-2]).float()
  z3 = torch.from_numpy(Vt[-3]).float()

  # AtA = A.T @ A
  # # # AtA = AtA + torch.eye(AtA.shape[0], dtype=AtA.dtype, device=AtA.device) * 1e-3  # Regularization for numerical stability
  # print("Shape of AtA:", AtA.shape)
  # # AtA = AtA.cpu().numpy()  # Convert to numpy for eigsh
  # print("Is NaN:", torch.isnan(AtA).any())  # Should be False
  # print("Is Inf:", torch.isinf(AtA).any())  # Should be False
  # print("Extremes:", AtA.min(), AtA.max())  # Look for extreme values
  # symmetric = torch.allclose(AtA, AtA.T, atol=1e-8)
  # print("Is symmetric:", symmetric)
  # # crude condition estimate
  # cond = AtA.abs().max() / AtA.abs().min()
  # print("Condition number estimate:", cond)

  # # eigenvalues, eigenvectors = torch.linalg.eigh(AtA)
  # # smallest_eigenvalues_indices = torch.argsort(eigenvalues)[:3]
  # # z1 = eigenvectors[:, smallest_eigenvalues_indices[0]]
  # # z2 = eigenvectors[:, smallest_eigenvalues_indices[1]]
  # # z3 = eigenvectors[:, smallest_eigenvalues_indices[2]]

  # eigenvalues, eigenvectors = eigsh(AtA.cpu().numpy(), k=3, which="SM")
  # eigenvectors = torch.from_numpy(eigenvectors).float()
  # z1 = eigenvectors[:, 0]
  # z2 = eigenvectors[:, 1]
  # z3 = eigenvectors[:, 2]
  # print("Eigenvalues:", eigenvalues)

  rotations = torch.stack([z1, z2, z3], dim=1).vsplit(m)
  initial_correction = None
  visual_rotations = []
  for i in range(len(rotations)):
    rotation = rotations[i].numpy()
    U, _, Vt = np.linalg.svd(rotation)
    rotation = U @ Vt  # Orthnormalized
    rotation = rotation.T
    if i == 0:
      initial_correction = inertial_rotations[0] @ rotation.T

    rotation = initial_correction @ rotation
    if mask[i]:
      visual_rotations.append(rotation)
    else:
      visual_rotations.append(None)

  return visual_rotations


def get_angle_difference(features1, features2, cameraMatrix, current_frame=None, previous_frame=None):
  kp1, des1 = features1
  kp2, des2 = features2

  # index_params, search_params = dict(algorithm=1, trees=5), dict(checks=50)
  # flann = cv.FlannBasedMatcher(index_params, search_params)
  # matches = flann.knnMatch(des1, des2, k=2)
  bf = cv.BFMatcher()
  matches12 = bf.knnMatch(des1, des2, k=2)
  matches21 = bf.knnMatch(des2, des1, k=2)

  reverse_map = {}
  for i, matchPairs in enumerate(matches21):
    if len(matchPairs) == 2:
      m, n = matchPairs
      if m.distance < 0.7 * n.distance:
        reverse_map[m.queryIdx] = m.trainIdx
  good_points_1, good_points_2 = [], []
  for i, matchPairs in enumerate(matches12):
    if len(matchPairs) == 2:
      m, n = matchPairs
      if m.distance < 0.7 * n.distance:
        if m.trainIdx in reverse_map and reverse_map[m.trainIdx] == m.queryIdx:
          gp1, gp2 = kp1[m.queryIdx].pt, kp2[m.trainIdx].pt
          good_points_1.append(gp1)
          good_points_2.append(gp2)

  good_points_1 = np.array(good_points_1)
  good_points_2 = np.array(good_points_2)

  if len(good_points_1) < 4:
    return (None, None, None)
  H, mask = cv.findHomography(good_points_1, good_points_2, cv.USAC_MAGSAC, 0.25)
  if H is None:
    return (None, None, None)
  inliers = np.sum(mask)
  inliers_dice = (2 * inliers) / (len(kp1) + len(kp2))

  if inliers < 50 or inliers_dice < 0.1:
    return (None, None, None)
  # if inliers < 10 or inliers_dice < 0.03:
  #   return (None, None, None)

  inlier_points_1 = good_points_1[mask.ravel() == 1]
  inlier_points_2 = good_points_2[mask.ravel() == 1]

  H, _ = cv.findHomography(inlier_points_1, inlier_points_2, 0)  # Least squares refinement

  extracted_rotation = np.linalg.inv(cameraMatrix) @ H @ cameraMatrix
  orthonormality = np.linalg.norm(extracted_rotation @ extracted_rotation.T - np.eye(3))

  U, _, Vt = np.linalg.svd(extracted_rotation)
  orthonormalized_rotation = U @ Vt

  if np.linalg.det(orthonormalized_rotation) < 0:
    return (None, None, None)

  # rotation = visual_rotation_to_global(orthonormalized_rotation)
  rotation = entire_transformation(orthonormalized_rotation)
  overlap_percent = get_homography_overlap_percent(rotation, cameraMatrix)

  if overlap_percent < 0.15:
    return (None, None, None)

  match_image = None
  if current_frame is not None:
    # match_image = cv.drawKeypoints(current_frame, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    match_image = current_frame.copy()
    if previous_frame is not None:
      # match_image = np.hstack(
      #   (
      #     cv.drawKeypoints(
      #       previous_frame,
      #       kp1,
      #       None,
      #       flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
      #     ),
      #     match_image,
      #   )
      # )
      # for i in range(len(good_points_1)):
      #   if mask[i] and np.random.rand() < 0.05:
      #     pt1 = tuple(map(int, good_points_1[i]))
      #     pt2 = tuple(map(int, good_points_2[i] + np.array([current_frame.shape[1], 0])))
      #     cv.line(match_image, pt1, pt2, (0, 0, 255), 1)

      # Draw currentframe on top of previous_frame, warped by the homography
      bestH = cameraMatrix @ orthonormalized_rotation @ np.linalg.inv(cameraMatrix)
      match_image = cv.warpPerspective(
        match_image,
        np.linalg.inv(bestH),
        (current_frame.shape[1], current_frame.shape[0]),
        flags=cv.INTER_LINEAR,
      )
      match_image = cv.addWeighted(match_image, 0.5, previous_frame, 0.5, 0)

    else:
      for i in range(len(good_points_1)):
        if mask[i]:
          pt1 = tuple(map(int, good_points_1[i]))
          pt2 = tuple(map(int, good_points_2[i]))
          cv.line(match_image, pt1, pt2, (0, 0, 255), 1)
      bestH = cameraMatrix @ orthonormalized_rotation @ np.linalg.inv(cameraMatrix)
      match_image = overlay_homography(match_image, bestH, cameraMatrix)

  extra_info = dict(
    overlap_percent=overlap_percent,
    orthonormality=orthonormality,
    inliers=inliers,
    inliers_dice=inliers_dice,
  )
  return rotation, match_image, extra_info


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Estimate angles from gyro data and video.")
  parser.add_argument("directory", type=str, help="Path to the directory containing the video and inertial data.")
  parser.add_argument(
    "--output_debug_video",
    action="store_true",
    help="Output a video for debugging purposes with visualisation of matches and angle estimation.",
  )
  parser.add_argument("--use_features_cache", action="store_true")
  parser.add_argument("--use_matches_cache", action="store_true")
  parser.add_argument("--naive_chaining", action="store_true")

  args = parser.parse_args()
  main(args)
