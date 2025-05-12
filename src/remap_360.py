import math
import torch
import line_profiler

def getFrameOutputVectors(output_width, output_height, device):
  y_angles = torch.linspace(-0.5, 0.5, output_height, device=device) * -torch.pi
  x_angles = torch.linspace(-1, 1, output_width, device=device) * torch.pi

  y_angles_grid, x_angles_grid = torch.meshgrid(y_angles, x_angles, indexing='ij')

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
  # NOTE: There is an unreproducible bug, where tensor sizes don't match. Loging valid_mask for when it happens
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