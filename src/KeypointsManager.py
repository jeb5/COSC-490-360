import math
import numpy as np
from scipy.spatial import ckdtree
from scipy.spatial.transform import Rotation as R


class KeypointsManager:
  def __init__(self, K):
    self.kpLocs = generate_fibonacci_sphere_points(K)
    self.tree = ckdtree.cKDTree(self.kpLocs)
    self.keypoints = [None] * K

  def add_potential_keypoint(self, rotation, features, index):
    rot_matrix = rotation.as_matrix()
    position = rot_matrix @ np.array([0, 1, 0])
    nearest_dist, nearest_idx = self.tree.query(position)
    self.keypoints[nearest_idx] = {
      "rotation": rotation,
      "features": features,
      "index": index,
    }

  def get_closest_keypoint(self, rotation):
    rot_matrix = rotation.as_matrix()
    position = rot_matrix @ np.array([0, 1, 0])
    nearest_dist, nearest_idx = self.tree.query(position)
    return self.keypoints[nearest_idx]


def generate_fibonacci_sphere_points(n):
  points = []
  phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

  for i in range(n):
    y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
    radius = math.sqrt(1 - y * y)  # radius at y

    theta = phi * i  # golden angle increment

    x = math.cos(theta) * radius
    z = math.sin(theta) * radius

    # rot = R.align_vectors([[0, 1, 0]], [[x, y, z]])[0]
    # points.append((np.array([x, y, z]), rot))

    points.append([x, y, z])

  return np.asarray(points)
