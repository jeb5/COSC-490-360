
import torch
import numpy as np
import cv2 as cv

# Maps are:
# [h, w, 2], where the last dimension is (x, y) (between -1 and 1)


# First apply map1 to an image, then apply map2 to the result
def combineRemappings(map1, map2):
  map1 = map1.clone().permute(2, 0, 1).unsqueeze(0) + 5
  map2 = map2.clone().unsqueeze(0)
  result = torch.nn.functional.grid_sample(map1, map2, mode='bilinear', padding_mode='zeros', align_corners=True)
  result = result.squeeze(0).permute(1, 2, 0) - 5
  return result

# map is [h, w, 2]
# image is [h, w, channels]


def torch_remap(map, image):
  image = image.clone().permute(2, 0, 1).unsqueeze(0)
  image_alpha = image[:, 3:4, :, :]
  image_matte = image[:, 0:3, :, :]
  map = map.clone().unsqueeze(0)

  matte_result = torch.nn.functional.grid_sample(image_matte, map, mode='bilinear', padding_mode='border', align_corners=True)
  alpha_result = torch.nn.functional.grid_sample(image_alpha, map, mode='bilinear', padding_mode='zeros', align_corners=True)

  result = torch.cat((matte_result, alpha_result), dim=1)

  return result.squeeze(0).permute(1, 2, 0)


def absoluteToRelative(map):
  relative_map = map.clone()
  (hw, hh) = map.shape[1] / 2.0, map.shape[0] / 2.0
  relative_map[..., 0] = (map[..., 0] - hw) / hw
  relative_map[..., 1] = (map[..., 1] - hh) / hh
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
  return absoluteToRelative(distorted)
