import numpy as np
import cv2 as cv
import spatialmedia
import spatialmedia.metadata_utils
import torch

# 100
GOPRO_CAMERA = (
	np.array([[1050.14, 0.0, 947.65],
           [0.0, 1057.60, 532.78],
            [0.0, 0.0, 1.0]]),
	np.array([0.0459, 0.0144, 0.00425, -0.01387])
)

# 100mm = 1920 pixels
# focal length = 35mm, so focal length = 35/100 * 1920 = 672
BLENDER_CAMERA = (
	np.array([[672.0, 0.0, 1920.0 / 2],
           [0.0, 672.0, 1080.0 / 2],
            [0.0, 0.0, 1.0]]),
	np.array([0.0, 0.0, 0.0, 0.0])
)

# 100mm = 1920 pixels
# focal length = 1050 pixels, so focal length = 1050/1920 * 100 = 54.68mm
BLENDER_CAMERA_2 = (
	np.array([[1050, 0.0, 1920.0 / 2],
           [0.0, 1050, 1080.0 / 2],
           [0.0, 0.0, 1.0]]),
	np.array([0.0, 0.0, 0.0, 0.0])
)


def add_transparent_image(background, foreground):

  fg = foreground.astype(float)
  bg = background.astype(float)

  fg_alpha = fg[:, :, 3] / 255.0
  bg_alpha = bg[:, :, 3] / 255.0

  fg_color = fg[:, :, :3] / 255.0
  bg_color = bg[:, :, :3] / 255.0

  output_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
  fg_alpha_3 = cv.merge((fg_alpha, fg_alpha, fg_alpha))

  fg_color = cv.multiply(fg_color, fg_alpha_3)
  bg_color = cv.multiply(1.0 - fg_alpha_3, bg_color)

  outImage = cv.add(fg_color, bg_color)
  outImage = cv.merge((outImage, output_alpha))
  outImage = outImage * 255.0

  return outImage.astype(np.uint8)


def BGRAToBGRAlphaBlack(image):
  if image.shape[2] != 4:
    raise ValueError("Image must have 4 channels")
  if image.dtype != np.uint8:
    raise ValueError("Image must be of type uint8")

  bgr = image[:, :, :3]
  alpha = image[:, :, 3]
  alpha = alpha.astype(float) / 255.0
  alpha = cv.merge((alpha, alpha, alpha))
  bgr = cv.multiply(bgr.astype(float), alpha)
  return bgr.astype(np.uint8)

def addSphericalMetadata(input_path, output_path):
	metadata = spatialmedia.metadata_utils.Metadata()
	metadata.video = spatialmedia.metadata_utils.generate_spherical_xml()
	def logging(message): print(message)
	spatialmedia.metadata_utils.inject_metadata(input_path, output_path, metadata, logging)


# def blenderCyclesPolynomialFisheyeUndistort(size, K):
#   k = [0, -0.439, -0.001, 0, 0]
#   sensor = 100.0  # mm
#   w, h = size
#   hw, hh = w / 2, h / 2
#   output = torch.zeros((h, w, 2), dtype=torch.float32)
#   for x_raw in range(w):
#     for y_raw in range(h):
#       x = ((x_raw - hw) / hw) * sensor
#       y = ((y_raw - hh) / hw) * sensor
#       r = np.sqrt(x ** 2 + y ** 2)
#       theta = k[0] + k[1] * r + k[2] * r ** 2 + k[3] * r ** 3 + k[4] * r ** 4
#       phi = np.acos(x / r)
#       out_x = np.cos(theta)
#       out_y = np.sin(theta) - np.cos(phi)
#       output[y_raw, x_raw, 0] = out_x * hw + hw
#       output[y_raw, x_raw, 1] = out_y * hw + hh
#   return output
