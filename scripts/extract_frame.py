import argparse
import cv2 as cv
import numpy as np
from angle_estimation import get_angle_difference
import helpers
from angle_estimation_helpers import get_features


def load_and_undistort_frame(video_path, frame_number, cam_matrix, cam_distortion):
  cap = cv.VideoCapture(video_path)
  cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
  ret, frame = cap.read()
  cap.release()
  if not ret:
    raise RuntimeError(f"Could not read frame {frame_number}")

  h, w = frame.shape[:2]
  if len(cam_distortion) > 0:
    new_cam_mat = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
      cam_matrix, cam_distortion, (w, h), None, None, 1, (w, h), 1
    )
    m1, m2 = cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, new_cam_mat, (w, h), cv.CV_32FC1)
    frame = cv.remap(frame, m1, m2, cv.INTER_LINEAR, borderValue=(255, 255, 255))
    # frame = cv.remap(frame, m1, m2, cv.INTER_LINEAR)
  return frame


def draw_keypoints(image, keypoints):
  return cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_matches(image1, kp1, image2, kp2, matches, mask):
  h1 = image1.shape[0]
  stacked = np.vstack((image1, image2)).copy()
  for i, m in enumerate(matches):
    if mask[i]:
      pt1 = tuple(map(int, kp1[m.queryIdx].pt))
      pt2 = tuple(map(int, kp2[m.trainIdx].pt))
      pt2 = (pt2[0], pt2[1] + h1)
      cv.line(stacked, pt1, pt2, (0, 255, 0), 1)
  return stacked


def match_keypoints(desc1, desc2):
  index_params = dict(algorithm=1, trees=5)
  search_params = dict(checks=50)
  flann = cv.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(desc1, desc2, k=2)

  good = []
  for m, n in matches:
    if m.distance < 0.7 * n.distance:
      good.append(m)
  return good


def main(directory, frame1, frame2, do_match):
  video_path = helpers.get_file_path_pack_dir(directory, "video")
  camera_info_path = helpers.get_file_path_pack_dir(directory, "camera_info")
  cam_matrix, cam_distortion = helpers.load_camera_info(camera_info_path)

  img1 = load_and_undistort_frame(video_path, frame1, cam_matrix, cam_distortion)
  img2 = load_and_undistort_frame(video_path, frame2, cam_matrix, cam_distortion)

  kp1, des1 = get_features(img1)
  kp2, des2 = get_features(img2)

  _, match_image, _ = get_angle_difference((kp1, des1), (kp2, des2), cam_matrix, current_frame=img1)

  cv.imwrite("temp/match_output.jpg", match_image)
  print("Saved image to temp/match_output.jpg")

  # if do_match:
  #   matches = match_keypoints(des1, des2)
  #   if len(matches) < 4:
  #     print("Not enough matches.")
  #     stacked = np.vstack((draw_keypoints(img1, kp1), draw_keypoints(img2, kp2)))
  #   else:
  #     pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
  #     pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
  #     H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 3.0)
  #     if H is None:
  #       print("Homography estimation failed.")
  #       stacked = np.vstack((draw_keypoints(img1, kp1), draw_keypoints(img2, kp2)))
  #     else:
  #       mask = mask.ravel().astype(bool)
  #       stacked = draw_matches(img1, kp1, img2, kp2, matches, mask)
  # else:
  #   out1 = draw_keypoints(img1, kp1)
  #   out2 = draw_keypoints(img2, kp2)
  #   stacked = np.vstack((out1, out2))

  # cv.imwrite("temp/sift_output.jpg", stacked)
  # print("Saved image to temp/sift_output.jpg")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("directory", type=str)
  parser.add_argument("frame1", type=int)
  parser.add_argument("frame2", type=int)
  parser.add_argument("--match", action="store_true", help="Draw inlier matches between frames using RANSAC")
  args = parser.parse_args()
  main(args.directory, args.frame1, args.frame2, args.match)
