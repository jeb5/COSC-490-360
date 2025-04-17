import cv2 as cv
import helpers
import numpy as np
import torch
import os
import remap


def main():
  np.set_printoptions(precision=3, suppress=True)
  os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
  cam_matrix = helpers.GOPRO_CAMERA[0]
  cam_distortion = helpers.GOPRO_CAMERA[1]

  size = (1920, 1080)
  # stmap = generateUniformSTMap(size)

  input_image = cv.imread("./input/test_pattern.png")
  input_image = cv.cvtColor(input_image, cv.COLOR_BGR2BGRA)
  input_image = torch.from_numpy(input_image).float()

  # newMat = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(cam_matrix, cam_distortion, size, None, None, 1, size, 1)
  m1, m2 = cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, cam_matrix, size, cv.CV_32F)
  undistort_map = torch.stack((torch.from_numpy(m1), torch.from_numpy(m2)), dim=-1)
  undistort_map = remap.absoluteToRelative(undistort_map)
  undistorted_image = remap.torch_remap(undistort_map, input_image)

  undistorted_image_cv = undistorted_image.numpy().astype(np.uint8)
  cv.imshow("undistorted Image", undistorted_image_cv)
  cv.imwrite("./output/undistorted_image.png", undistorted_image_cv)
  cv.waitKey(0)
  cv.destroyAllWindows()

  # inputK = cam_matrix
  # size = (int(1920 * 0.7), int(1080 * 0.7))
  # outputK = np.array([[1050, 0.0, 1920 / 2],
  #                     [0.0, 1050, 1080 / 2],
  #                     [0.0, 0.0, 1.0]])

  distort_map = remap.getFisheyeDistortionMap(size, cam_matrix, cam_distortion)
  distorted_image = remap.torch_remap(distort_map, input_image)
  distorted_image_cv = distorted_image.numpy().astype(np.uint8)
  cv.imshow("distorted Image", distorted_image_cv)
  cv.imwrite("./output/distorted_image.png", distorted_image_cv)
  cv.waitKey(0)
  cv.destroyAllWindows()

def generateUniformSTMap(size):
  w, h = size
  max = 1.0
  step = max / w
  grid_x = torch.linspace(0, max - step, w)
  grid_y = torch.linspace(max - step, 0, h)
  grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
  stmap = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x), torch.ones_like(grid_x)], dim=0)
  return toCVImage(stmap)


def applySTMap(stmap, image):
  if image.shape[2] != 4:
    image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
  if stmap.dtype != np.float32:
    raise ValueError("STMap must be of type float32")
  # remapped_image = np.zeros_like(image, dtype=image.dtype)
  (h, w) = stmap.shape[:2]
  # for u in range(w):
  #   for v in range(h):
  #     _, y, x, a = stmap[v, u]
  #     image_x = int(x * w)
  #     image_y = int((1 - y) * h)
  #     if a != 0:
  #       remapped_image[v, u] = image[image_y, image_x]
  #       remapped_image[v, u][3] = a * image[image_y, image_x][3]
  stmap_map = torch.from_numpy(stmap)
  stmap_map = stmap_map.unsqueeze(0)
  stmap_alpha = stmap_map[:, :, :, 3:4]
  stmap_map = (stmap_map[:, :, :, 1:3] - 0.5) * 2
  # Swap x and y
  temp = stmap_map[:, :, :, 0].clone()
  stmap_map[:, :, :, 0] = stmap_map[:, :, :, 1]
  stmap_map[:, :, :, 1] = temp

  torch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
  result = torch.nn.functional.grid_sample(torch_image, stmap_map, mode='bilinear', padding_mode='zeros', align_corners=False)
  result = result.permute(0, 2, 3, 1)
  result[:, :, :, 3:4] = stmap_alpha * result[:, :, :, 3:4]
  result = result.squeeze(0).int().numpy().astype(np.uint8)

  return result


def remapTransborder(image, mapX, mapY):
  if image.shape[2] != 4:
    image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
  # Overscan
  ospix = 10
  image_overscanned = np.pad(image, ((ospix, ospix), (ospix, ospix), (0, 0)), mode='edge')
  # Make the first ospix rows and columns have alpha = 0
  image_overscanned[:ospix, :, 3] = 0
  image_overscanned[-ospix:, :, 3] = 0
  image_overscanned[:, :ospix, 3] = 0
  image_overscanned[:, -ospix:, 3] = 0
  mapX = mapX + 3
  mapY = mapY + 3
  mapX = np.clip(mapX, 0, image_overscanned.shape[1] - 1)
  mapY = np.clip(mapY, 0, image_overscanned.shape[0] - 1)

  remapped = cv.remap(image_overscanned, mapX, mapY, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

  return remapped


# def apply_fisheye_effect(img, K, d):
#   w, h = img.shape[1], img.shape[0]

#   indices = np.array(np.meshgrid(range(h), range(w))).T
#   indices = indices.reshape(np.prod(img.shape[:2]), -1).astype(np.float32)

#   Kinv = np.linalg.inv(K)
#   indices1 = np.zeros_like(indices, dtype=np.float32)
#   for i in range(len(indices)):
#     x, y = indices[i]
#     indices1[i] = (Kinv @ np.array([[x], [y], [1]])).squeeze()[:2]
#   indices1 = indices1[np.newaxis, :, :]

#   in_indices = cv2.fisheye.distortPoints(indices1, K, d)
#   indices, in_indices = indices.squeeze(), in_indices.squeeze()

#   distorted_img = np.zeros_like(img)
#   for i in range(len(indices)):
#     x, y = indices[i]
#     ix, iy = in_indices[i]
#     if (ix < img.shape[0]) and (iy < img.shape[1]):
#       distorted_img[int(ix), int(iy)] = img[int(x), int(y)]

#   return distorted_img

# The above code (taken from stackoverflow) seems stupid


def getFisheyeDistortionMap(outputSize, K, D):
  (w, h) = outputSize
  indices = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing="ij"))
  # vv Watch out for indices becoming larger than max int
  indices = indices.permute(1, 2, 0).reshape(-1, 2)
  indices = indices.numpy().astype(np.float32)[:, np.newaxis, :]
  distorted = cv.fisheye.undistortPoints(indices, K, D, None, K).reshape(w, h, 2)
  mapX = distorted[:, :, 0].T
  mapY = distorted[:, :, 1].T
  return mapX, mapY

def toCVImage(image):
  image = image.permute(1, 2, 0).numpy()
  image = cv.cvtColor(image, cv.COLOR_RGBA2BGRA)
  return image


if __name__ == "__main__":
  main()
