import argparse
import cv2 as cv
import numpy as np


def main(args):
  image = cv.imread(args.image_path)
  if image is None:
    print(f"Error: Could not read image from {args.image_path}")
    return
  # Convert to grayscale
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  blurred = cv.GaussianBlur(gray, (11, 11), 0)
  cv.imshow("Blurred", blurred)
  cv.waitKey(0)
  ret, otsued = cv.threshold(blurred, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
  cv.imshow("Otsu Threshold", otsued * 255)
  cv.waitKey(0)
  image_thresholded = otsued
  image_closed = cv.morphologyEx(image_thresholded, cv.MORPH_CLOSE, kernel=np.ones((80, 80), np.uint8))
  cv.imshow("Closed Image", image_closed * 255)
  cv.waitKey(0)
  # Use contours to find the horizon

  # This method seems to fail at medium-high altitudes, when there is a layer of clouds sitting above the horizion. Determinining whether a detected line is the horizon or the top of clouds seems not trivial.
  # Perhaps something more sophisticated, such as a semantic segementation model could help


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Detect horizon in an image.")
  parser.add_argument("image_path", type=str, help="Path to the input image file.")
  args = parser.parse_args()
  main(args)
