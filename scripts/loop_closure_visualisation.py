import math
import numpy as np
import matplotlib.pyplot as plt
import torch


def Rz(alpha):
  return torch.tensor(
    [[math.cos(alpha), -math.sin(alpha), 0], [math.sin(alpha), math.cos(alpha), 0], [0, 0, 1]], dtype=torch.float32
  )


def Ry(beta):
  return torch.tensor([[math.cos(beta), 0, math.sin(beta)], [0, 1, 0], [-math.sin(beta), 0, math.cos(beta)]], dtype=torch.float32)


def Rx(gamma):
  return torch.tensor(
    [[1, 0, 0], [0, math.cos(gamma), -math.sin(gamma)], [0, math.sin(gamma), math.cos(gamma)]], dtype=torch.float32
  )


def make_rotation_loop(num_steps=100, pitch_delta_deg=0.0, yaw_delta_deg=360.0):
  rotations = []
  for i in range(num_steps):
    yaw = math.radians(yaw_delta_deg) * (i / (num_steps - 1))
    pitch = math.radians(pitch_delta_deg) * (i / (num_steps - 1))
    roll = 0
    rot_mat = Rz(yaw) @ Rx(pitch) @ Ry(roll)
    rotations.append(rot_mat.numpy())
  return rotations


def plot_rotation_path(rotations, color, title):
  fig = plt.figure(figsize=(4, 4))
  ax = fig.add_subplot(projection="3d")
  ax.set_box_aspect([1, 1, 1])
  ax.set_title(title)

  vectors = np.array([R @ np.array([0, 1, 0]) for R in rotations])
  xs, ys, zs = vectors[:, 0], vectors[:, 1], vectors[:, 2]
  ax.plot(xs, ys, zs, c=color, marker=".")

  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.set_zlim(-1, 1)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_zticklabels([])
  return fig


# First plot: pure yaw loop
yaw_only_rotations = make_rotation_loop()
fig1 = plot_rotation_path(yaw_only_rotations, color="#f07d0a", title="Loop Closure")

# Second plot: yaw + pitch loop
yaw_pitch_rotations = make_rotation_loop(pitch_delta_deg=16.0, yaw_delta_deg=350.0)
fig2 = plot_rotation_path(yaw_pitch_rotations, color="#f07d0a", title="Without loop closure")

plt.show()
