
import torch
import numpy as np
import cv2 as cv
import line_profiler

import helpers

# Maps are:
# [h, w, 2], where the last dimension is (x, y) (between -1 and 1)


# First apply map1 to an image, then apply map2 to the result
# TODO: Combined remappins might be a fool's errand (How to deal with out-of-bounds pixels?)
def combineRemappings(map1, map2):
  map1 = map1.clone().permute(2, 0, 1).unsqueeze(0)
  map2 = map2.clone().unsqueeze(0)
  result = torch.nn.functional.grid_sample(map1, map2, mode='bilinear', padding_mode='zeros', align_corners=True)
  result = result.squeeze(0).permute(1, 2, 0)
  return result

# map is [h, w, 2]
# image is [h, w, channels]


@line_profiler.profile
def torch_remap(map, image):
  has_alpha = image.shape[2] == 4
  image = image.clone().permute(2, 0, 1).unsqueeze(0)
  image_matte = image[:, 0:3, :, :]
  image_alpha = image[:, 3:4, :, :]
  if not has_alpha:
    image_alpha = torch.full_like(image_matte[:, 0:1, :, :], 255)
  map = map.clone().unsqueeze(0)

  matte_result = torch.nn.functional.grid_sample(image_matte, map, mode='bilinear', padding_mode='reflection', align_corners=True) # padding mode border unavailable on MPS
  alpha_result = torch.nn.functional.grid_sample(image_alpha, map, mode='bilinear', padding_mode='zeros', align_corners=True)

  result = torch.cat((matte_result, alpha_result), dim=1).squeeze(0).permute(1, 2, 0)
  if not has_alpha:
    result = helpers.BGRAToBGRAlphaBlack_torch(result)

  return result


@line_profiler.profile
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
