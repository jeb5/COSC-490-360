MAX_OBSERVATION_CACHE_SIZE = 100000


class ObservationManager:
  def __init__(self, feature_manager, validator_func, orientation_estimation_func):
    self.feature_manager = feature_manager
    self.validator_func = validator_func
    self.orientation_estimation_func = orientation_estimation_func
    self.cached_observations = {}
    self.cache_expiration_queue = []

  def get_observations_in_window(self, start_frame, end_frame):
    frame_pairs = [(i, j) for i in range(start_frame, end_frame) for j in range(start_frame, i)]
    return self.get_observations(frame_pairs)

  def get_observations(self, frame_pairs):
    for i, j in frame_pairs:
      observation = self.__get_observation(i, j)
      if observation is not None:
        yield observation

  def __get_observation(self, i, j):
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
    return result
