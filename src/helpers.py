import numpy as np
import cv2 as cv
import torch
import torchvision
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# 100
GOPRO_CAMERA = (
	np.array([[1050.14, 0.0, 947.65],
           [0.0, 1057.60, 532.78],
            [0.0, 0.0, 1.0]]),
	np.array([0.0459, 0.0144, 0.00425, -0.01387])
)

# 100mm = 1920 pixels
# focal length = 1050 pixels, so focal length = 1050/1920 * 100 = 54.6875mm
BLENDER_CAMERA = (
	np.array([[1050, 0.0, 1920.0 / 2],
           [0.0, 1050, 1080.0 / 2],
           [0.0, 0.0, 1.0]]),
	np.array([0.0, 0.0, 0.0, 0.0])
)

BLENDER_CAMERA_WITH_FISHEYE = (
	BLENDER_CAMERA[0],
  GOPRO_CAMERA[1]
)

font_path = 'src/assets/roboto_mono.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.monospace'] = fm.FontProperties(fname=font_path).get_name()

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

# Images are BGRA [h, w, 4]


def paste_cv(background, foreground, x, y):
  fgx1, fgy1 = max(-x, 0), max(-y, 0)
  fgx2, fgy2 = min(x + foreground.shape[1], background.shape[1]) - x, min(y + foreground.shape[0], background.shape[0]) - y
  if fgx1 >= fgx2 or fgy1 >= fgy2:
    return background
  bgx1, bgy1 = x + fgx1, y + fgy1
  bgx2, bgy2 = x + fgx2, y + fgy2
  background[bgy1:bgy2, bgx1:bgx2] = foreground[fgy1:fgy2, fgx1:fgx2]
  return background

def add_transparent_image_torch(background, foreground):
  fg_alpha = foreground[:, :, 3] / 255.0
  bg_alpha = background[:, :, 3] / 255.0

  fg_color = foreground[:, :, :3] / 255.0
  bg_color = background[:, :, :3] / 255.0

  output_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
  fg_alpha_3 = torch.stack((fg_alpha, fg_alpha, fg_alpha), dim=-1)
  fg_color = fg_color * fg_alpha_3
  bg_color = bg_color * (1 - fg_alpha_3)

  outImage = fg_color + bg_color
  outImage = torch.cat((outImage, output_alpha.unsqueeze(-1)), dim=-1)
  outImage = outImage * 255.0
  return outImage


def display_torch_image(image, window_name="Image"):
  image_cv = BGRAToBGRAlphaBlack_torch(image).cpu().numpy().astype(np.uint8)
  cv.imshow(window_name, image_cv)
  cv.waitKey(0)
  cv.destroyAllWindows()

def centerCrop(image, target_width, target_height):
  top, left = (image.shape[0] - target_height) // 2, (image.shape[1] - target_width) // 2
  cropped = image.permute(2, 0, 1)
  cropped = torchvision.transforms.functional.crop(cropped, top, left, target_height, target_width)
  cropped = cropped.permute(1, 2, 0)
  return cropped


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

# Image is BGRA [h, w, 4]


def BGRAToBGRAlphaBlack_torch(image):
  if image.shape[2] != 4:
    raise ValueError("Image must have 4 channels")
  if image.dtype != torch.float32:
    raise ValueError("Image must be of type float32")

  bgr = image[:, :, :3]
  alpha = image[:, :, 3]
  alpha = alpha / 255.0
  alpha = torch.stack((alpha, alpha, alpha), dim=-1)
  bgr = bgr * alpha
  return bgr

def get_device():
  device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
  if device == 'cpu':
    print("Warning: Using CPU for remapping, which may be slow")
  return device


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
