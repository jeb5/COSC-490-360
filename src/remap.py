
import torch
import numpy as np
import cv2 as cv
import helpers


def torch_remap(map, image):
  has_alpha = image.shape[2] == 4
  image = image.clone().permute(2, 0, 1).unsqueeze(0)
  image_matte = image[:, 0:3, :, :]
  image_alpha = image[:, 3:4, :, :]
  if not has_alpha:
    image_alpha = torch.full_like(image_matte[:, 0:1, :, :], 255)
  map = map.clone().unsqueeze(0)

  matte_result = torch.nn.functional.grid_sample(
    image_matte, map, mode="bilinear", padding_mode="reflection", align_corners=True
  )  # padding mode border unavailable on MPS
  alpha_result = torch.nn.functional.grid_sample(image_alpha, map, mode="bilinear", padding_mode="zeros", align_corners=True)

  result = torch.cat((matte_result, alpha_result), dim=1).squeeze(0).permute(1, 2, 0)
  if not has_alpha:
    result = helpers.BGRAToBGRAlphaBlack_torch(result)

  return result


def absoluteToRelative(map, source_size):
  relative_map = torch.empty_like(map)
  (hw, hh) = source_size[0] / 2.0, source_size[1] / 2.0
  relative_map[..., 0] = map[..., 0] * (1 / hw) - 1
  relative_map[..., 1] = map[..., 1] * (1 / hh) - 1
  return relative_map


def getFisheyeDistortionMap(outputSize, K, D):
  (w, h) = outputSize
  indices = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing="ij"))
  indices = indices.permute(1, 2, 0).reshape(-1, 2)
  # vv Watch out for indices becoming larger than max int
  if indices.shape[0] > 2**31:
    raise ValueError("Too many pixels in image")
  indices = indices.numpy().astype(np.float32)[:, np.newaxis, :]
  distorted = cv.fisheye.undistortPoints(indices, K, D, None, K).reshape(w, h, 2)
  distorted = torch.from_numpy(distorted).transpose(0, 1)
  return absoluteToRelative(distorted, outputSize)

def custom_fisheye_undistort_map(K, D, image_size, new_K):
  w, h = image_size
  fx, fy = new_K[0, 0], new_K[1, 1]
  cx, cy = new_K[0, 2], new_K[1, 2]
  old_fx, old_fy = K[0, 0], K[1, 1]
  old_cx, old_cy = K[0, 2], K[1, 2]
  map1, map2 = np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
  for i in range(h):
    for j in range(w):
      x = (j - cx) / fx
      y = (i - cy) / fy
      r = np.sqrt(x**2 + y**2)
      theta = np.arctan(r)
      theta_d = theta * (1 + D[0] * theta**2 + D[1] * theta**4 + D[2] * theta**6 + D[3] * theta**8)
      scale = theta_d / r if r != 0 else 1.0
      x_distorted = x * scale
      y_distorted = y * scale
      map1[i, j] = x_distorted * old_fx + old_cx
      map2[i, j] = y_distorted * old_fy + old_cy
  return map1, map2