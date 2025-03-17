import numpy as np
import cv2 as cv

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

  return outImage
