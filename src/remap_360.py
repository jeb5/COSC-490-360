import math
import torch

def getFrameOutputVectors(output_width, output_height, device):
  y_angles = torch.linspace(-0.5, 0.5, output_height, device=device) * -torch.pi
  x_angles = torch.linspace(-1, 1, output_width, device=device) * torch.pi

  y_angles_grid, x_angles_grid = torch.meshgrid(y_angles, x_angles, indexing='ij')

  Ca = torch.cos(x_angles_grid)
  Cb = torch.cos(y_angles_grid)
  Sa = torch.sin(x_angles_grid)
  Sb = torch.sin(y_angles_grid)

  output_vectors = torch.stack([Cb * Sa, Sb, Ca * Cb], dim=-1)
  return output_vectors


def remapping360_torch(image_width, image_height, rotation, focal_length, output_vectors):
  # This function no longer uses flattening (Which can result in too-long vectors with lengths over the integer limit)
  # Also the A[mask] = B[mask] pattern is no longer used, as it occasionally results unreproducible errors

  yaw, pitch, roll = rotation.as_euler("ZXY")
  camera_rotation_matrix = (Ry(-yaw) @ Rx(-pitch) @ Rz(-roll)).to(output_vectors.device)

  v_transformed = torch.einsum('hwc, cd -> hwd', output_vectors, camera_rotation_matrix)
  v_transformed[:, :, 1] = -v_transformed[:, :, 1]  # Flip Y axis
  valid_mask = v_transformed[:, :, 2] > 0  # Only keep forward-facing pixels
  valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, 3).type_as(v_transformed)  # Expand mask to match v_transformed shape
  v_transformed *= focal_length / v_transformed[:, :, 2:3]
  projected = valid_mask * v_transformed + (1 - valid_mask) * -100000

  mapX = projected[:, :, 0] + (image_width / 2)
  mapY = projected[:, :, 1] + (image_height / 2)

  # return mapX.cpu().numpy(), mapY.cpu().numpy()

  # Relative ([-1, 1]) coordinates
  mapX = mapX * (2 / image_width) - 1
  mapY = mapY * (2 / image_height) - 1

  map = torch.stack((mapX, mapY), dim=-1)  # [h, w, 2]
  return map

# ZYX = YAW, PITCH, ROLL
# Yaw around Z, Pitch around Y, Roll around X

def Rz(alpha):
  return torch.tensor([[math.cos(alpha), -math.sin(alpha), 0],
                      [math.sin(alpha), math.cos(alpha), 0],
                      [0, 0, 1]])


def Ry(beta):
  return torch.tensor([[math.cos(beta), 0, math.sin(beta)],
                      [0, 1, 0],
                      [-math.sin(beta), 0, math.cos(beta)]])


def Rx(gamma):
  return torch.tensor([[1, 0, 0],
                      [0, math.cos(gamma), -math.sin(gamma)],
                      [0, math.sin(gamma), math.cos(gamma)]])