import os
import pickle
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
  global_rotation_to_visual,
  load_features,
  overlay_homography,
  visual_rotation_to_global,
  inertials_from_csv,
)
import helpers
import progressbar as pb
import sys
import signal
from video_writer import VideoWriter
from scipy.spatial.transform import Rotation as R


def main(args):
  input_video = cv.VideoCapture(args.video_path)
  if not input_video.isOpened():
    print("Error opening video file")
    return

  input_size = (
    int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT)),
  )
  input_framerate = int(input_video.get(cv.CAP_PROP_FPS))
  (start_frame, end_frame) = (
    args.start_frame,
    (args.end_frame if args.end_frame >= 0 else int(input_video.get(cv.CAP_PROP_FRAME_COUNT) - 1)),
  )

  cam_matrix, cam_distortion = helpers.GOPRO_CAMERA if args.gopro else helpers.BLENDER_CAMERA
  new_cam_mat = (
    cv.fisheye.estimateNewCameraMatrixForUndistortRectify(cam_matrix, cam_distortion, input_size, None, None, 1, input_size, 1)
    if args.gopro
    else cam_matrix
  )
  m1, m2 = (
    cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, new_cam_mat, input_size, cv.CV_32FC1)
    if args.gopro
    else (None, None)
  )

  if args.output_video is not None:
    output_video = VideoWriter(args.output_video, input_framerate, input_size)
  if args.output_debug_video is not None:
    output_debug_video = VideoWriter(args.output_debug_video, input_framerate, input_size)
    # IF output_debug_video was disabled, we *could* save time by not undistorting the image, only the feature locations

  def cleanup():
    input_video.release()
    if args.output_video is not None:
      if output_debug_video.did_write():
        output_video.save_video()
        print(f"Output video saved to {args.output_video}")
    if args.output_debug_video is not None:
      if output_debug_video.did_write():
        output_debug_video.save_video()
        print(f"Debug video saved to {args.output_debug_video}")
    print("Done.")

  def interupt_handler(signum, frame):
    cleanup()
    sys.exit(0)

  signal.signal(signal.SIGINT, interupt_handler)

  bar = pb.ProgressBar(
    max_value=end_frame - start_frame + 1,
    widgets=["Reading inertial data: ", pb.GranularBar()],
  )

  inertial_rotations = [
    R.from_euler("ZXY", [xyz[2], xyz[0], xyz[1]], degrees=True).as_matrix()
    for i, xyz in bar(inertials_from_csv(args, start_frame, end_frame))
  ]

  features_cache_path = f"{args.video_path}.features.data"
  frame_features = load_features(features_cache_path)
  if frame_features is not None:
    print(f"Loaded features from cache: {features_cache_path}")
  else:
    frame_features = []
    bar = pb.ProgressBar(
      max_value=len(inertial_rotations),
      widgets=["Finding SIFT features: ", pb.Percentage(), " ", pb.GranularBar()],
    )
    for i in bar(range(end_frame - start_frame + 1)):
      frame = start_frame + i
      input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
      ret, image = input_video.read()
      image_undistorted = cv.remap(image, m1, m2, cv.INTER_LINEAR) if args.gopro else image
      frame_features.append(get_features(image_undistorted))
    cache_features(frame_features, features_cache_path)

  # solve_rotations(frame_features, new_cam_mat, inertial_rotations)

  visual_rotations = []
  if True:
    visual_rotations = chain_rotations(
      frame_features,
      inertial_rotations,
      new_cam_mat,
      input_video,
      start_frame,
      end_frame,
      args,
      output_video if args.output_video is not None else None,
      output_debug_video if args.output_debug_video is not None else None,
      m1,
      m2,
    )
  else:
    visual_rotations = solve_rotations(frame_features, new_cam_mat, inertial_rotations, args)
  # # f1, f2 = 83, 71
  # j, i = 71, 83
  # # j, i = 60, 61

  # # f1, f2 = 21, 20
  # input_video.set(cv.CAP_PROP_POS_FRAMES, j)
  # j_frame = input_video.read()[1]
  # input_video.set(cv.CAP_PROP_POS_FRAMES, i)
  # i_frame = input_video.read()[1]
  # j_features = frame_features[j - start_frame]
  # i_features = frame_features[i - start_frame]
  # visual_rotation_change, match_image, extra_info = get_angle_difference(j_features, i_features, new_cam_mat, i_frame, j_frame)
  # # backwards, _, _ = get_angle_difference(features2, features1, new_cam_mat, frame1, frame2)
  # # H1 = new_cam_mat @ global_rotation_to_visual(visual_rotation_change) @ np.linalg.inv(new_cam_mat)
  # # H2 = new_cam_mat @ global_rotation_to_visual(backwards) @ np.linalg.inv(new_cam_mat)

  # # visual_rotation_change = visual_rotation_change.T
  # # v_zxy = R.from_matrix(visual_rotation_change).as_euler("ZXY", degrees=True)
  # # visual_rotation_change = R.from_euler("ZXY", [-v_zxy[0], -v_zxy[1], -v_zxy[2]], degrees=True).as_matrix()
  # # visual_rotation_change = np.linalg.inv(visual_rotation_change)  # We want i->j not j->i
  # print(f"Extra info: {extra_info}")
  # # intertial_rot = inertial_rotations[f1] @ np.linalg.inv(inertial_rotations[f2])
  # print(f"Intertial start: {R.from_matrix(inertial_rotations[j]).as_euler('ZXY', degrees=True)}")
  # print(f"Intertial end: {R.from_matrix(inertial_rotations[i]).as_euler('ZXY', degrees=True)}")
  # # intertial_rot = inertial_rotations[f2] @ np.linalg.inv(inertial_rotations[f1])
  # intertial_rot = np.linalg.inv(inertial_rotations[j]) @ inertial_rotations[i]
  # difference = np.linalg.inv(visual_rotation_change) @ intertial_rot
  # angle_diff = np.linalg.norm(R.from_matrix(difference).as_rotvec(degrees=True))
  # print(f"Angle difference: {angle_diff:.2f} degrees")
  # # print visual rotation change zxy
  # visual_zxy = R.from_matrix(visual_rotation_change).as_euler("ZXY", degrees=True)
  # print(f"Visual rotation change: {visual_zxy[0]:.2f}y {visual_zxy[1]:.2f}p {visual_zxy[2]:.2f}r")
  # # print inertial rotation change zxy
  # inertial_zxy = R.from_matrix(intertial_rot).as_euler("ZXY", degrees=True)
  # print(f"Inertial rotation change: {inertial_zxy[0]:.2f}y {inertial_zxy[1]:.2f}p {inertial_zxy[2]:.2f}r")
  # # Show the match image
  # if match_image is not None:
  #   cv.imshow("Match Image", match_image)
  #   cv.waitKey(0)
  #   cv.destroyAllWindows()

  # Outputtting visual rotation (in xyz order)
  with open(args.output_angle_path, "w") as csvfile:
    for frame, visual_rotation in enumerate(visual_rotations):
      visual_zxy = R.from_matrix(visual_rotation).as_euler("ZXY", degrees=True)
      csvfile.write(
        "{},{},{},{}\n".format(
          frame,
          visual_zxy[1],
          visual_zxy[2],
          visual_zxy[0],
        )
      )
  cleanup()


def chain_rotations(
  frame_features,
  inertial_rotations,
  new_cam_mat,
  input_video,
  start_frame,
  end_frame,
  args,
  output_video,
  output_debug_video,
  m1,
  m2,
):
  visual_rotations = []

  bar = pb.ProgressBar(
    max_value=len(inertial_rotations),
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
  )
  for i in bar(range(end_frame - start_frame + 1)):
    frame = start_frame + i
    input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
    ret, image = input_video.read()
    if not ret:
      raise Exception(f"Failed to read frame {frame} from video.")
    image_undistorted = cv.remap(image, m1, m2, cv.INTER_LINEAR) if args.gopro else image

    visual_rotation_change, match_image = None, None
    if i == 0:
      visual_rotations.append(inertial_rotations[0])
      match_image = image_undistorted.copy()
    else:
      visual_rotation_change, match_image, _ = get_angle_difference(
        frame_features[i - 1], frame_features[i], new_cam_mat, image_undistorted
      )
      if visual_rotation_change is None:
        raise Exception(f"Homography estimation failed at frame {frame}")
      visual_rotations.append(visual_rotations[i - 1] @ visual_rotation_change)

    visual_zxy = R.from_matrix(visual_rotations[i]).as_euler("ZXY", degrees=True)
    inertial_zxy = R.from_matrix(inertial_rotations[i]).as_euler("ZXY", degrees=True)

    if args.output_debug_video is not None:
      angleText = ""
      angleText += f"Frame {i + 1}/{len(inertial_rotations)}"
      angleText += f"\nInertial:\n{inertial_zxy[0]:6.2f}y {inertial_zxy[1]:6.2f}p {inertial_zxy[2]:6.2f}r"
      angleText += f"\nVisual:\n{visual_zxy[0]:6.2f}y {visual_zxy[1]:6.2f}p {visual_zxy[2]:6.2f}r"
      if visual_rotation_change is not None:
        visual_zxy_change = R.from_matrix(visual_rotation_change).as_euler("ZXY", degrees=True)
        angleText += f"\nChange:\n{visual_zxy_change[0]:6.2f}y {visual_zxy_change[1]:6.2f}p {visual_zxy_change[2]:6.2f}r"
      vector_plot = generate_rotation_histories_plot(
        [
          {"name": "Visual", "colour": "#42a7f5", "data": visual_rotations[: i + 1]},
          {"name": "Inertial", "colour": "#f07d0a", "data": inertial_rotations[: i + 1]},
        ],
        extra_text=angleText,
        # extra_rot=visual_rotation_change,
      )
      (x, y) = (
        match_image.shape[1] - vector_plot.shape[1],
        match_image.shape[0] - vector_plot.shape[0],
      )
      match_image = helpers.paste_cv(match_image, vector_plot, x, y)

      # image_debug = add_text_to_image(match_image, angleText)
      output_debug_video.write_frame_opencv(match_image)
    if args.output_video is not None:
      output_video.write_frame_opencv(image)
  return visual_rotations


def find_matches(sift_features, cameraMatrix, inertial_rotations):
  frame_relations_picture = np.zeros((len(sift_features) * 2, len(sift_features), 3), dtype=np.uint8)
  matches = []
  n = len(sift_features)
  max_value = n * (n - 1) // 2
  bar = pb.ProgressBar(
    max_value=max_value,
    widgets=["Finding matches: ", pb.Percentage(), " ", pb.GranularBar(), " ", pb.ETA()],
    redirect_stdout=True,
  ).start()
  idx = 0
  total_angle_diff = 0.0
  for i in range(len(sift_features)):
    for j in range(0, i):
      # for j in range(max(i - 1, 0), i):
      match_rot, _, extra_info = get_angle_difference(sift_features[i], sift_features[j], cameraMatrix)

      # angle_diff =

      intertial_rotation_change = np.linalg.inv(inertial_rotations[j]) @ inertial_rotations[i]
      intertial_overlap_percent = get_homography_overlap_percent(intertial_rotation_change, cameraMatrix)
      # frame_relations_picture[i * 2 + 1, j, 1] = intertial_overlap_percent * 255
      if match_rot is not None:
        matches.append((i, j, match_rot))
        frame_relations_picture[i * 2, j, 0] = 255
        frame_relations_picture[i * 2, j, 2] = (min(extra_info["inliers_dice"] * 4, 1)) * 255

        intertial_rot = np.linalg.inv(inertial_rotations[i]) @ inertial_rotations[j]
        difference = np.linalg.inv(match_rot) @ intertial_rot
        angle_diff = np.linalg.norm(R.from_matrix(difference).as_rotvec(degrees=True))
        frame_relations_picture[i * 2 + 1, j, 1] = min(angle_diff / 10, 1) * 255
        total_angle_diff += angle_diff
        if angle_diff > 2:
          print("Large angle difference detected:")
          print(f"  Frame {i} to {j}: {angle_diff:.2f} degrees")
          print(f"  Inliers: {extra_info['inliers']}, Dice: {extra_info['inliers_dice']:.2f}")
          print(f"  Orthonormality: {extra_info['orthonormality']:.2f}")

      idx += 1
      bar.update(idx)
  bar.finish()
  print(f"Average angle difference: {total_angle_diff / len(matches):.2f} degrees")
  print(f"Total matches found: {len(matches)}")
  cv.imwrite("temp/frame_relations_new.png", frame_relations_picture)
  return matches


def solve_rotations(sift_features, cameraMatrix, inertial_rotations, args):
  # Take list of frame features (SIFT lists)
  # For each frame, get with ALL previous frames (n^2, or more precisely, n(n-1)/2)
  # For each match pair, estimate rotation using homography. If not enough points, skip.
  # We will make a note of any frames that had no matches. (OR frames with no connection to the first frame. Union-Find?)
  # We then construct a matrix A using only the frames that are inter-matched.
  # To solve Az = 0, we construct A^TA, and take the 3 eigenvectors corresponding to the 3 smallest eigenvalues. This is equivilent to taking the 3 right singular vectors corresponding to the 3 smallest singular values of A
  # Now given 3 values of z, we reconstruct the rotation matrices R^{i} for each frame i.
  # These will be rotation matrices in a common coordinate frame, however not the world coordinate frame.
  # If we know the world coordinate frame of the first frame, we can rotate all matricies to the world coordinate frame.
  # sift_features = sift_features[:50]  # Limit to first 50 frames for testing
  matches_cache_path = f"{args.video_path}.matches.data"
  matches = []
  if os.path.exists(matches_cache_path) and False:
    print(f"Loading matches from cache: {matches_cache_path}")
    matches = pickle.load(open(matches_cache_path, "rb"))
  else:
    print("Finding matches...")
    matches = find_matches(sift_features, cameraMatrix, inertial_rotations)
    pickle.dump(matches, open(matches_cache_path, "wb"))

  l, m = len(matches), len(sift_features)  # noqa: E741
  A = torch.zeros((l * 3, m * 3), dtype=torch.float32)
  for idx, (i, j, match_rot) in enumerate(matches):
    A[idx * 3 : idx * 3 + 3, i * 3 : i * 3 + 3] = -torch.from_numpy(np.linalg.inv(match_rot))
    # A[idx * 3 : idx * 3 + 3, i * 3 : i * 3 + 3] = -torch.from_numpy(match_rot)
    A[idx * 3 : idx * 3 + 3, j * 3 : j * 3 + 3] = torch.eye(3, dtype=torch.float32)
  U, S, Vt = torch.linalg.svd(A)
  z1 = Vt[-1]
  z2 = Vt[-2]
  z3 = Vt[-3]
  rotations = torch.stack([z1, z2, z3], dim=1).vsplit(m)
  initial_correction = None
  visual_rotations = []
  for i in range(len(rotations)):
    rotation = rotations[i].numpy()
    U, _, Vt = np.linalg.svd(rotation)
    rotation = U @ Vt  # Orthnormalized
    rotation = rotation.T
    if i == 0:
      initial_correction = rotation.T

    rotation = initial_correction @ rotation
    visual_rotations.append(rotation)
    # zxy = R.from_matrix(rotation).as_euler("ZXY", degrees=True)
    # print(f"Frame {i}: {zxy[0]:6.2f}y {zxy[1]:6.2f}p {zxy[2]:6.2f}r")
  generate_rotation_histories_plot(
    [
      {"name": "Visual", "colour": "#42a7f5", "data": visual_rotations},
      {"name": "Inertial", "colour": "#f07d0a", "data": inertial_rotations},
    ],
    interactive=True,
  )

  # Concatenate

  return []


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

  # if inliers < 50 or inliers_dice < 0.1:
  # return (None, None, None)
  if inliers < 4 or inliers_dice < 0.04:
    return (None, None, None)

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

  if overlap_percent < 0.1:
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
  parser.add_argument("video_path", type=str, help="Path to the input video file.")
  parser.add_argument("gyro_csv_path", type=str, help="Path to the gyro CSV file.")
  parser.add_argument("output_angle_path", type=str, help="Path to the output angle CSV file.")
  parser.add_argument("--output_video", type=str, help="Path to the output video file.")
  parser.add_argument("--output_debug_video", type=str, help="Path to the debug output video file.")
  parser.add_argument("--start_frame", type=int, default=0, help="Start frame for processing.")
  parser.add_argument("--end_frame", type=int, default=-1, help="End frame for processing.")
  parser.add_argument("--gopro", action="store_true", help="Use GoPro camera settings.")
  args = parser.parse_args()
  main(args)
