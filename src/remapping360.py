#!/usr/bin/env python3
# import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import cv2 as cv
import line_profiler

# Not used
def remapping360(output_width, output_height, image_width, image_height, yaw, pitch, roll, focal_length):
  half_output_width, half_output_height = output_width / 2, output_height / 2
  half_image_width, half_image_height = image_width / 2, image_height / 2

  camera_rotation_matrix = XYZRotationMatrix(pitch, yaw, roll)
  inv_camera_rotation_matrix = torch.linalg.inv(camera_rotation_matrix)

  mapX = torch.zeros((output_height, output_width), dtype=torch.float32)
  mapY = torch.zeros((output_height, output_width), dtype=torch.float32)

  output_vectors = torch.zeros(
    (output_height, output_width, 3), dtype=torch.float32)

  # Generate output vectors
  for j in range(output_height):
    output_pitch = -((j - half_output_height) / output_height) * torch.pi
    for i in range(output_width):
      output_yaw = ((i - half_output_width) / output_width) * 2 * torch.pi
      # R = YXZRotationMatrix(output_yaw, output_pitch, 0)
      # v = R.dot(torch.array([0, 0, 1]))
      Ca = math.cos(output_yaw)
      Cb = math.cos(output_pitch)
      Sa = math.sin(output_yaw)
      Sb = math.sin(output_pitch)
      v = torch.array([Cb * Sa, -Sb, Ca * Cb])
      output_vectors[j, i] = v

  # Use output vectors to generate remapping
  last_progress_report = -1
  for j in range(output_height):
    progress = int((j / output_height) * 100)
    if progress != last_progress_report:
      print(f"Progress: {progress}%")
      last_progress_report = progress
    for i in range(output_width):
      v = output_vectors[j, i]

      v = inv_camera_rotation_matrix.dot(v)
      if v[2] < 0:
        # Pointing away from image plane
        mapX[j, i] = -100000  # Arbitrary values to indicate invalid pixel
        mapY[j, i] = -100000
      else:
        v = v * (focal_length / v[2])

        mapX[j, i] = v[0] + half_image_width
        mapY[j, i] = v[1] + half_image_height

  return mapX, mapY


def getFrameOutputVectors(output_width, output_height, device):
  y_angles = torch.linspace(-0.5, 0.5, output_height, device=device) * -torch.pi
  x_angles = torch.linspace(-1, 1, output_width, device=device) * torch.pi

  y_angles_grid, x_angles_grid = torch.meshgrid(y_angles, x_angles, indexing='ij')

  # TODO: Increase speed by meshgriding the cosine and sines??
  # TODO: Increase speed by precalculating this part, and reusing it for subsequent frames

  Ca = torch.cos(x_angles_grid)
  Cb = torch.cos(y_angles_grid)
  Sa = torch.sin(x_angles_grid)
  Sb = torch.sin(y_angles_grid)

  output_vectors = torch.stack([Cb * Sa, Sb, Ca * Cb], dim=-1)
  output_vectors = output_vectors.reshape(-1, 3)
  return output_vectors


@line_profiler.profile
def remapping360_torch(output_width, output_height, image_width, image_height, yaw, pitch, roll, focal_length, output_vectors, device):

  half_image_width, half_image_height = image_width / 2, image_height / 2
  camera_rotation_matrix = (Ry(-yaw) @ Rx(-pitch) @ Rz(roll)).to(device)
  inv_camera_rotation_matrix = torch.linalg.inv(camera_rotation_matrix)

  # Flatten for batch processing

  v_transformed = torch.mm(inv_camera_rotation_matrix, output_vectors.T).T
  valid_mask = v_transformed[:, 2] > 0  # Only keep forward-facing pixels
  v_transformed *= focal_length / v_transformed[:, 2:3]
  # v_transformed[:, 0] += 12
  # v_transformed[:, 0] -= 2

  # Undo the flattening (see note below)
  v_transformed = v_transformed.reshape(output_height, output_width, 3)
  v_transformed[:, :, 1] = -v_transformed[:, :, 1]  # Flip Y axis
  valid_mask = valid_mask.reshape(output_height, output_width)

  mapX = torch.full((output_height, output_width), -100000,
                    dtype=torch.float32, device=device)
  mapY = torch.full((output_height, output_width), -100000,
                    dtype=torch.float32, device=device)

  # count valid pixels
  valid_pixels = torch.sum(valid_mask)
  print(f"Valid pixels: {valid_pixels}")
  mapX[valid_mask] = v_transformed[valid_mask][:, 0] + half_image_width
  mapY[valid_mask] = v_transformed[valid_mask][:, 1] + half_image_height

  # RIP
  # Here lies many hours wasted by an infurating bug
  # Relevant tensors were once flattened, thus having lengthes so large that masking them with a boolean tensor was impossible
  # The masking behaved very strangely, and eventually I noticed that all the problems started at a suspicious index: 16,777,216
  # I've therefore unflattened the tensors, and now everything works perfectly

  return mapX.cpu().numpy(), mapY.cpu().numpy()

# ZYX = YAW, PITCH, ROLL
# Yaw around Z, Pitch around Y, Roll around X


@line_profiler.profile
def Rz(alpha):
  return torch.tensor([[math.cos(alpha), -math.sin(alpha), 0],
                      [math.sin(alpha), math.cos(alpha), 0],
                      [0, 0, 1]])


@line_profiler.profile
def Ry(beta):
  return torch.tensor([[math.cos(beta), 0, math.sin(beta)],
                      [0, 1, 0],
                      [-math.sin(beta), 0, math.cos(beta)]])


@line_profiler.profile
def Rx(gamma):
  return torch.tensor([[1, 0, 0],
                      [0, math.cos(gamma), -math.sin(gamma)],
                      [0, math.sin(gamma), math.cos(gamma)]])

# map: width x height x 2, where map[x,y, 0] says which x coordinate from the original image should be used for pixel (x,y)
# def fisheyeUndistort(map, )
  # fisheye undistort takes the x,y locations given by the map, and replaces them with what should be there, if the image were undistorted
