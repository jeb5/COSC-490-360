import cv2 as cv
import helpers
import numpy as np
import torch
import os


def main():
  np.set_printoptions(precision=3, suppress=True)
  os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
  cam_matrix = helpers.GOPRO_CAMERA[0]
  cam_distortion = helpers.GOPRO_CAMERA[1]
  size = cam_matrix.shape[1], cam_matrix.shape[0]

  stmap = generateUniformSTMap((1920, 1080))
  distored_stmap = applyFisheyeEffect(stmap, cam_matrix, cam_distortion)
  return

  # newMat = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(cam_matrix, cam_distortion, size, None, None, 1, size, 1)
  # m1, m2 = cv.fisheye.initUndistortRectifyMap(cam_matrix, cam_distortion, None, newMat, size, cv.CV_32F)

  m1, m2 = cv.fisheye.initInv

  undistorted_stmap = remapTransborder(stmap, m1, m2)
  input_image = cv.imread("./input/balloon_frame_1.png")

  result = applySTMap(undistorted_stmap, input_image)


def generateUniformSTMap(size):
  w, h = size
  max = np.finfo(np.float32).max
  max = 1
  grid_x = torch.linspace(0, max, w)
  grid_y = torch.linspace(max, 0, h)
  grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
  stmap = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x), torch.ones_like(grid_x)], dim=0)
  return toCVImage(stmap)


def applySTMap(stmap, image):
  if image.shape[2] != 4:
    image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
  if stmap.shape != image.shape:
    raise ValueError("STMap and image must have the same shape")
  # STMap should be float32
  if stmap.dtype != np.float32:
    raise ValueError("STMap must be of type float32")
  remapped_image = np.zeros_like(image, dtype=image.dtype)
  (h, w) = image.shape[:2]
  cv.imshow("STMap", stmap)
  cv.imwrite("./output/stmap.exr", stmap)
  cv.waitKey(0)
  cv.imshow("Image", image)
  cv.waitKey(0)
  # For STMap pixels with 0 alpha, output can be black, 0 alpha
  # For STMap pixels with (0, 1) alpha, output will be the remapped pixel, with alpha of STMap * alpha of image
  for u in range(w):
    for v in range(h):
      _, y, x, a = stmap[v, u]
      image_x = int(x * (w - 1))
      image_y = int((1 - y) * (h - 1))
      # if u == int(w / 2):
      # print(f"STMap: {stmap[v, u]}")
      # print(f"Image: {image[image_y, image_x]}")
      if a != 0:
        remapped_image[v, u] = image[image_y, image_x]
        remapped_image[v, u][3] = a * image[image_y, image_x][3]
  cv.imshow("Remapped", remapped_image)
  cv.imwrite("./output/remapped.png", remapped_image)
  cv.waitKey(0)
  return remapped_image


def remapTransborder(image, mapX, mapY):
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
  cv.imwrite("./output/overscanned.exr", image_overscanned)
  mapX = np.clip(mapX, 0, image_overscanned.shape[1] - 1)
  mapY = np.clip(mapY, 0, image_overscanned.shape[0] - 1)

  remapped = cv.remap(image_overscanned, mapX, mapY, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

  return remapped


def apply_fisheye_effect(img, K, d):
  w, h = img.shape[1], img.shape[0]

  indices = np.array(np.meshgrid(range(h), range(w))).T
  indices = indices.reshape(np.prod(img.shape[:2]), -1).astype(np.float32)

  Kinv = np.linalg.inv(K)
  indices1 = np.zeros_like(indices, dtype=np.float32)
  for i in range(len(indices)):
    x, y = indices[i]
    indices1[i] = (Kinv @ np.array([[x], [y], [1]])).squeeze()[:2]
  indices1 = indices1[np.newaxis, :, :]

  in_indices = cv2.fisheye.distortPoints(indices1, K, d)
  indices, in_indices = indices.squeeze(), in_indices.squeeze()

  distorted_img = np.zeros_like(img)
  for i in range(len(indices)):
    x, y = indices[i]
    ix, iy = in_indices[i]
    if (ix < img.shape[0]) and (iy < img.shape[1]):
      distorted_img[int(ix), int(iy)] = img[int(x), int(y)]

  return distorted_img

# The above code (taken from stackoverflow) seems stupid


def applyFisheyeEffect(image, K, D):
  (w, h) = image.shape[1], image.shape[0]
  k_inv = torch.linalg.inv(K)
  indices = np.array(np.meshgrid(range(h), range(w))).T
  print(indices)


def toCVImage(image):
  image = image.permute(1, 2, 0).numpy()
  image = cv.cvtColor(image, cv.COLOR_RGBA2BGRA)
  print(image.dtype)
  return image


if __name__ == "__main__":
  main()
