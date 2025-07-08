import os
import pickle
import cv2 as cv
from matplotlib import pyplot as plt
import shapely
import numpy as np


def overlay_homography(image, homography, cameraMatrix):
  #   H = cameraMatrix @ global_rotation_to_visual(rotation) @ np.linalg.inv(cameraMatrix)

  w, h = cameraMatrix[0, 2] * 2, cameraMatrix[1, 2] * 2
  np_coords = np.array(
    [
      [0, 0, 1],
      [w, 0, 1],
      [w, h, 1],
      [0, h, 1],
      [0, 0, 1],
    ]
  )
  transformed_coords = (homography @ np_coords.T).T
  transformed_coords = transformed_coords / np.abs(transformed_coords[:, 2:3])  # Normalize by z
  transformed_coords = transformed_coords[:, :2]  # Drop z coordinate

  cv.polylines(
    image,
    [np.int32(transformed_coords)],
    isClosed=True,
    color=(0, 255, 0),
    thickness=2,
  )
  return image


def get_features(image, lower_bound=200):
  threshold_multiplier = 1.0
  kp, des = None, None
  while True:
    sift = cv.SIFT_create(
      nfeatures=5000,
      contrastThreshold=0.04 * threshold_multiplier,
      edgeThreshold=10 * threshold_multiplier,
    )
    kp, des = sift.detectAndCompute(image, None)
    if len(kp) < lower_bound:
      threshold_multiplier *= 0.8
      if threshold_multiplier < 0.2:
        raise ValueError("Failed to find enough keypoints in image")
    else:
      break
  return (kp, des)


def visual_rotation_to_global(rotation):
  # # Convert from y-up to z-up
  # reorder_mat = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
  # rotation = reorder_mat.T @ visual_rotation @ reorder_mat
  # # Reverse roll
  # zxy = R.from_matrix(rotation).as_euler("ZXY", degrees=True)
  # rotation = R.from_euler("ZXY", [zxy[0], zxy[1], -zxy[2]], degrees=True).as_matrix()
  # return rotation
  return np.array(
    [
      [rotation[0, 0], rotation[2, 0], -rotation[1, 0]],
      [rotation[0, 2], rotation[2, 2], -rotation[1, 2]],
      [-rotation[0, 1], -rotation[2, 1], rotation[1, 1]],
    ]
  )
  # return rotation


def entire_transformation(rotation):
  # reorder_mat = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
  # rotation = reorder_mat.T @ rotation @ reorder_mat
  # zxy = R.from_matrix(rotation).as_euler("ZXY", degrees=True)
  # rotation = R.from_euler("ZXY", [-zxy[0], -zxy[1], zxy[2]], degrees=True).as_matrix()
  # rotation = rotation.T
  # return rotation
  return np.array(
    [
      [rotation[0, 0], rotation[2, 0], -rotation[1, 0]],
      [rotation[0, 2], rotation[2, 2], -rotation[1, 2]],
      [-rotation[0, 1], -rotation[2, 1], rotation[1, 1]],
    ]
  )


def global_rotation_to_visual(rotation):
  # # Reverse roll
  # zxy = R.from_matrix(global_rotation).as_euler("ZXY", degrees=True)
  # rotation = R.from_euler("ZXY", [zxy[0], zxy[1], -zxy[2]], degrees=True).as_matrix()
  # # Convert from z-up to y-up
  # reorder_mat = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
  # rotation = reorder_mat @ rotation @ reorder_mat.T
  # return rotation
  return np.array(
    [
      [rotation[0, 0], -rotation[2, 0], rotation[1, 0]],
      [-rotation[0, 2], rotation[2, 2], -rotation[1, 2]],
      [rotation[0, 1], -rotation[2, 1], rotation[1, 1]],
    ]
  )


def get_homography_overlap_percent(rotation, cameraMatrix):
  np.set_printoptions(precision=3, suppress=True)
  H = cameraMatrix @ global_rotation_to_visual(rotation) @ np.linalg.inv(cameraMatrix)
  w, h = cameraMatrix[0, 2] * 2, cameraMatrix[1, 2] * 2
  np_coords = np.array(
    [
      [0, 0, 1],
      [w, 0, 1],
      [w, h, 1],
      [0, h, 1],
      [0, 0, 1],
    ]
  )
  normal_coords = np_coords[:, :2]
  transformed_coords = (H @ np_coords.T).T
  transformed_coords = transformed_coords / np.abs(transformed_coords[:, 2:3])  # Normalize by z
  transformed_coords = transformed_coords[:, :2]  # Drop z coordinate

  normal_rect = shapely.Polygon(normal_coords)
  transformed_rect = shapely.Polygon(transformed_coords)
  transformed_linear_ring = shapely.LinearRing(transformed_coords)
  if not transformed_linear_ring.is_ccw:
    return 0.0

  overlap = 0.0
  try:
    overlap = normal_rect.intersection(transformed_rect).area
  except shapely.errors.GEOSException:
    pass
  normal_area = normal_rect.area
  return overlap / normal_area


# ZYX = YAW, PITCH, ROLL
# Yaw around Z, Pitch around Y, Roll around X


def Rz(alpha):
  return np.array(
    [
      [np.cos(alpha), -np.sin(alpha), 0],
      [np.sin(alpha), np.cos(alpha), 0],
      [0, 0, 1],
    ]
  )


def Ry(beta):
  return np.array(
    [
      [np.cos(beta), 0, np.sin(beta)],
      [0, 1, 0],
      [-np.sin(beta), 0, np.cos(beta)],
    ]
  )


def Rx(gamma):
  return np.array(
    [
      [1, 0, 0],
      [0, np.cos(gamma), -np.sin(gamma)],
      [0, np.sin(gamma), np.cos(gamma)],
    ]
  )


# Plots rotations histories in a z-up 3D space plot
def generate_rotation_histories_plot(rotation_histories, extra_text=None, extra_rot=None, interactive=False):
  text_fig = None
  if extra_text is not None:
    text_fig = plt.figure(figsize=(3, 1.4))
    ax_text = text_fig.add_subplot()
    ax_text.axis("off")
    ax_text.text(
      0,
      1.0,
      extra_text,
      fontsize=12,
      ha="left",
      va="top",
      wrap=True,
      color="black",
      transform=ax_text.transAxes,
      fontfamily="monospace",
      fontweight="bold",
    )

    text_fig.tight_layout()

  # Main 3D plot
  main_fig = plt.figure(figsize=(3, 3))
  ax_main = main_fig.add_subplot(projection="3d")
  ax_main.set_box_aspect(aspect=None, zoom=1)
  for rotation_history in rotation_histories:
    name = rotation_history["name"]
    colour = rotation_history["colour"]
    rotations = rotation_history["data"]
    # Get mask to filter out None values
    mask = [rot is not None for rot in rotations]
    mask = np.array(mask)
    rotations = [rot if rot is not None else np.eye(3) for rot in rotations]
    rotations = np.array(rotations)
    # Python version (old)
    # vectors = [rotation @ np.array([0, 1, 0]) for rotation in rotations]
    # Numpy version (new)
    vectors = rotations @ np.array([0, 1, 0])
    # Apply mask, setting None values to NaN
    vectors = np.where(mask[:, None], vectors, np.nan)

    vector_history = np.array(vectors)
    xs, ys, zs = vector_history[:, 0], vector_history[:, 1], vector_history[:, 2]
    ax_main.quiver(
      0,
      0,
      0,
      xs[-1],
      ys[-1],
      zs[-1],
      length=1,
      normalize=True,
      color=colour,
      arrow_length_ratio=0.2,
    )
    ax_main.plot(xs, ys, zs, c=colour, marker=".", label=name)

  if extra_rot is not None:
    extra_vector = extra_rot @ np.array([0, 1, 0])
    ax_main.quiver(
      0,
      0,
      0,
      extra_vector[0],
      extra_vector[1],
      extra_vector[2],
      length=1,
      normalize=True,
      color="red",
      arrow_length_ratio=0.2,
    )

  ax_main.set_xlabel("X", labelpad=-10)
  ax_main.set_ylabel("Y", labelpad=-10)
  ax_main.set_zlabel("Z", labelpad=-10)
  ax_main.set_xlim(-1, 1)
  ax_main.set_ylim(-1, 1)
  ax_main.set_zlim(-1, 1)
  ax_main.legend()
  ax_main.set_xticklabels([])
  ax_main.set_yticklabels([])
  ax_main.set_zticklabels([])

  if interactive:
    main_fig.show()
    plt.show(block=True)

  main_image = figure_to_cv_image(main_fig)
  if text_fig is None:
    return main_image
  text_image = figure_to_cv_image(text_fig)
  return np.vstack((text_image, main_image))


def figure_to_cv_image(fig):
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height(physical=True)
  plt.close(fig)
  image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
  image = image.reshape(h, w, 4)
  image = image[:, :, 1:4]
  image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

  return image




def serialize_keypoints(kp):
  return [
    (
      kp.pt[0],
      kp.pt[1],
      kp.size,
      kp.angle,
      kp.response,
      kp.octave,
      kp.class_id,
    )
    for kp in kp
  ]


def deserialize_keypoints(serialized_kp):
  return [
    cv.KeyPoint(x, y, size, angle, response, octave, class_id)
    for (x, y, size, angle, response, octave, class_id) in serialized_kp
  ]


def cache_features(features, cache_path):
  print(f"Caching features to: {cache_path}")
  with open(cache_path, "wb") as f:
    serialized_frame_features = [(serialize_keypoints(kp), des.tolist()) for (kp, des) in features]
    pickle.dump(serialized_frame_features, f)


def load_features(cache_path):
  print(f"Loading features from cache: {cache_path}")
  if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
      frame_features_serialized = pickle.load(f)
      frame_features = [(deserialize_keypoints(kp), np.asarray(des, dtype=np.float32)) for (kp, des) in frame_features_serialized]
      return frame_features
  else:
    print("Cache miss")
    return []


def load_matches(cache_path):
  print(f"Loading matches from cache: {cache_path}")
  if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
      matches = pickle.load(f)
      return matches
  else:
    print("Cache miss")
    return []


def cache_matches(matches, cache_path):
  print(f"Caching matches to: {cache_path}")
  with open(cache_path, "wb") as f:
    pickle.dump(matches, f)
