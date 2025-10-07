import numpy as np
import cv2 as cv

from FeatureManager import FeatureManager

MAX_OBSERVATION_CACHE_SIZE = 100000


class ObservationManager:
  def __init__(self, feature_manager: FeatureManager, validator_func, orientation_estimation_func, data_manager):
    self.feature_manager = feature_manager
    self.validator_func = validator_func
    self.orientation_estimation_func = orientation_estimation_func
    self.cached_observations = {}
    self.cache_expiration_queue = []
    self.data_manager = data_manager

  def get_observations_in_window(self, start_frame, end_frame):
    frame_pairs = [(i, j) for i in range(start_frame, end_frame) for j in range(start_frame, i)]
    return self.get_observations(frame_pairs)

  def get_observations(self, frame_pairs):
    for i, j in frame_pairs:
      observation = self.get_observation(i, j)
      if observation is not None:
        yield observation

  def get_observation(self, i, j, validate=True):
    i, j = min(i, j), max(i, j)  # Enforce i <= j
    if (i, j) in self.cached_observations:
      return self.cached_observations[(i, j)]

    fi = self.feature_manager.detect_features(i)
    fj = self.feature_manager.detect_features(j)
    matches = self.feature_manager.get_matches(fi, fj)
    relative_orientation, estimation_info = self.orientation_estimation_func(matches)
    result = None
    if self.validator_func(estimation_info):
      result = (i, j, relative_orientation)
    self.cached_observations[(i, j)] = result
    self.cache_expiration_queue.append((i, j))
    if len(self.cached_observations) >= MAX_OBSERVATION_CACHE_SIZE:
      oldest_pair = self.cache_expiration_queue.pop(0)
      del self.cached_observations[oldest_pair]
    if validate and result is None:
      return None
    return (i, j, relative_orientation)

  def generate_observation_image(self):
    # Get maximum and minimum indicies in cached observations
    if not self.cached_observations:
      return None
    min_index = min(i for i, _ in self.cached_observations.keys())
    max_index = max(j for _, j in self.cached_observations.keys())
    image = np.zeros((max_index - min_index + 1, max_index - min_index + 1, 3), dtype=np.uint8)
    for i, j in self.cached_observations.keys():
      rot = self.cached_observations[(i, j)]
      if rot is not None:
        color = (0, 255, 0)  # Green for valid observations
      else:
        color = (0, 0, 255)  # Red for invalid observations
      image[j - min_index, i - min_index] = color
    return image

  def show_interactive_observation_image(self):
    window_name = "Observation explorer"
    min_index = min(i for i, _ in self.cached_observations.keys())
    left = self.generate_observation_image()
    (lh, lw) = left.shape[:2]
    scale = 1000 / lh
    left = cv.resize(left, (int(left.shape[1] * scale), int(left.shape[0] * scale)), interpolation=cv.INTER_NEAREST)
    right = np.zeros_like(left)
    cv.namedWindow(window_name)
    current_pair = None

    def render():
      if current_pair is not None:
        nonlocal right
        print(f"Current pair: {current_pair}")
        right = self.draw_matches(*current_pair, True)
        right_rescale = (lh * scale) / right.shape[0]
        right = cv.resize(
          right, (int(right.shape[1] * right_rescale), int(right.shape[0] * right_rescale)), interpolation=cv.INTER_LINEAR
        )

      cv.imshow(window_name, np.hstack((left, right)))

    def on_mouse(event, x, y, flags, param):
      if event == cv.EVENT_LBUTTONDOWN:
        (px, py) = (x // scale, y // scale)
        if px < 0 or px >= lw or py < 0 or py >= lh:
          return
        print(f"Clicked on pixel ({px}, {py})")
        nonlocal current_pair
        current_pair = (px + min_index, py + min_index)
        print(f"Current pair: {current_pair}")
        render()

    render()
    cv.setMouseCallback(window_name, on_mouse)
    while True:
      if cv.waitKey(20) & 0xFF == 27:  # ESC to exit
        break
    cv.destroyWindow(window_name)

  def draw_matches(self, i, j, stacked=False):
    i, j = min(i, j), max(i, j)  # Enforce i <= j
    print(f"Drawing matches for frames {i} and {j}")
    framei = self.data_manager.get_frame(i, undistort=True)
    framej = self.data_manager.get_frame(j, undistort=True)
    fi = self.feature_manager.detect_features(i)
    fj = self.feature_manager.detect_features(j)
    matches = self.feature_manager.get_matches(fi, fj)
    pts1, pts2 = matches
    kptsi, _ = fi
    kptsj, _ = fj
    relative_orientation, estimation_info = self.orientation_estimation_func(matches)
    homography_mask = estimation_info.get("inlier_mask") if estimation_info else [False] * len(pts1)

    framei = cv.drawKeypoints(framei, kptsi, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if stacked:
      framej = cv.drawKeypoints(framej, kptsj, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      stacked_image = np.vstack((framei, framej))
      frame_height = framej.shape[0]

      for k, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        pt1 = tuple(map(int, pt1))
        pt2 = (int(pt2[0]), int(pt2[1] + frame_height))
        color = (0, 255, 0) if homography_mask[k] else (0, 0, 255)
        cv.line(stacked_image, pt1, pt2, color, 1)
      cv.putText(stacked_image, f"Frame {i}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
      cv.putText(stacked_image, f"Frame {j}", (10, 30 + frame_height), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
      return stacked_image
    else:
      for k, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        color = (0, 255, 0) if homography_mask[k] else (0, 0, 255)
        cv.line(framei, pt1, pt2, color, 1)
      cv.putText(framei, f"Frame {i}->{j}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
      return framei