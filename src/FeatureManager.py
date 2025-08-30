import numpy as np
from DataManager import DataManager
import cv2 as cv


def get_ORB_features(frame):
  orb = cv.ORB_create(nfeatures=2000)
  keypoints, descriptors = orb.detectAndCompute(frame, None)
  return keypoints, descriptors


def get_SIFT_features(frame):
  threshold_multiplier = 1.0
  while True:
    sift = cv.SIFT_create(
      nfeatures=5000,
      contrastThreshold=0.04 * threshold_multiplier,
      edgeThreshold=10 * threshold_multiplier,
    )
    kp, des = sift.detectAndCompute(frame, None)
    if len(kp) > 200 or threshold_multiplier < 0.15:
      return kp, des
    threshold_multiplier *= 0.8


def get_SURF_features(frame):
  threshold_multiplier = 1.0
  while True:
    surf = cv.xfeatures2d.SURF_create(
      hessianThreshold=200 * threshold_multiplier,
    )
    kp, des = surf.detectAndCompute(frame, None)
    if len(kp) > 200 or threshold_multiplier < 0.15:
      return kp, des
    threshold_multiplier *= 0.8


MAX_FEATURE_CACHE_SIZE = 1000


class FeatureManager:
  def __init__(
    self,
    data_manager: DataManager,
    feature_type,
    ratio_test_threshold=0.7,
    orientation_filter=True,
    cross_check=False,
    undistort_shortcut=True,
  ):
    self.data_manager = data_manager
    self.undistort_shortcut = undistort_shortcut
    self.undistort_required = len(data_manager.distortion_coefficients) > 0
    assert feature_type in ["ORB", "SIFT", "SURF"], "Unsupported feature type"
    self.feature_type = feature_type
    self.feature_detector = {
      "ORB": get_ORB_features,
      "SIFT": get_SIFT_features,
      "SURF": get_SURF_features,
    }[feature_type]
    self.intrinsic_matrix, self.distortion_coefficients, _ = data_manager.get_camera_info()
    self.ratio_test_threshold = ratio_test_threshold
    self.orientation_filter = orientation_filter
    self.cross_check = cross_check
    self.cached_features = {}
    self.cache_expiration_queue = []

  def detect_features(self, frame_number):
    if frame_number in self.cached_features:
      return self.cached_features[frame_number]

    frame = self.data_manager.get_frame(frame_number, undistort=(self.undistort_required and (not self.undistort_shortcut)))
    kps, des = self.feature_detector(frame)
    if self.undistort_required and self.undistort_shortcut:
      points = np.array([kp.pt for kp in kps], dtype=np.float32)
      undistorted_points = cv.undistortPoints(
        points, self.intrinsic_matrix, self.distortion_coefficients, None, None, self.intrinsic_matrix
      )
      for i, kp in enumerate(undistorted_points):
        kps[i].pt = tuple(kp[0])

    self.cached_features[frame_number] = (kps, des)
    self.cache_expiration_queue.append(frame_number)
    if len(self.cached_features) >= MAX_FEATURE_CACHE_SIZE:
      oldest_frame = self.cache_expiration_queue.pop(0)
      del self.cached_features[oldest_frame]

    return kps, des

  def get_matches(self, features1, features2):
    kp1, des1 = features1
    kp2, des2 = features2

    matcher = None
    if self.feature_type == "ORB":
      matcher = cv.BFMatcher(cv.NORM_HAMMING)
    elif self.feature_type in ["SIFT", "SURF"]:
      # For SIFT and SURF, we use FLANN-based matcher
      index_params = dict(algorithm=1, trees=5)
      search_params = dict(checks=50)
      matcher = cv.FlannBasedMatcher(index_params, search_params)

    all_matches_12 = matcher.knnMatch(des1, des2, k=2)

    matches = []
    for matchPairs in all_matches_12:
      if len(matchPairs) <= 2:
        m, n, *_ = matchPairs
        if m.distance < self.ratio_test_threshold * n.distance:
          matches.append(m)

    if self.cross_check:
      # Filter out matches which are not symmetric
      all_matches_21 = matcher.knnMatch(des2, des1, k=2)
      reverse_map = {}
      for matchPairs in all_matches_21:
        if len(matchPairs) <= 2:
          m, n, *_ = matchPairs
          if m.distance < self.ratio_test_threshold * n.distance:
            reverse_map[m.queryIdx] = m.trainIdx
      matches = [m for m in matches if m.trainIdx in reverse_map and reverse_map[m.trainIdx] == m.queryIdx]

    if self.orientation_filter:
      # Filter matches based on keypoint orientation
      angle_bins = [[] for _ in range(360 // 5)]
      for match in matches:
        angle_difference = kp1[match.queryIdx].angle - kp2[match.trainIdx].angle
        if angle_difference < 0:
          angle_difference += 360
        bin_number = int(angle_difference // 5)
        angle_bins[bin_number].append(match)
      max_bin_number = 0
      for i, bin in enumerate(angle_bins):
        if len(bin) > len(angle_bins[max_bin_number]):
          max_bin_number = i
      bin_num = max_bin_number
      bin_num_min_one = (bin_num - 1 + len(angle_bins)) % len(angle_bins)
      bin_num_plus_one = (bin_num + 1) % len(angle_bins)
      matches = angle_bins[bin_num] + angle_bins[bin_num_min_one] + angle_bins[bin_num_plus_one]

    points1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    return points1, points2
