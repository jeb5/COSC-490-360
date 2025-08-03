import cv2
import numpy as np
import sys


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


# === Hardcoded Camera Parameters ===
K = np.array([[1050.14, 0.0, 947.65], [0.0, 1057.60, 532.78], [0.0, 0.0, 1.0]])
D = np.array([[0.0459], [0.0144], [0.00425], [-0.01387]])

# === Script Starts Here ===
if len(sys.argv) < 2:
  print("Usage: python undistort_fisheye.py <path_to_image>")
  sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)

if img is None:
  print(f"Failed to load image: {img_path}")
  sys.exit(1)

h, w = img.shape[:2]

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
undistorted_cv = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("Undistorted (OpenCV)", undistorted_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("undistorted_cv.png", undistorted_cv)

map1, map2 = custom_fisheye_undistort_map(K, D, (w, h), new_K)
undistorted_custom = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("Undistorted (Custom)", undistorted_custom)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("undistorted_custom.png", undistorted_custom)
