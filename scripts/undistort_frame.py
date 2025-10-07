import sys
import cv2
import numpy as np

if len(sys.argv) < 2:
  print("Usage: python undistort_fisheye.py <image_path>")
  sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
if img is None:
  print("Could not load image:", img_path)
  sys.exit(1)

# Camera intrinsics
K = np.array([[1045, 0.0, 960], [0.0, 1054, 540], [0.0, 0.0, 1.0]], dtype=np.float64)

D = np.array([0.045, 0.012, 0.003, -0.004], dtype=np.float64)

h, w = img.shape[:2]

# Estimate new camera matrix
new_K = K.copy()
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

# Use BORDER_TRANSPARENT to leave unmapped pixels undefined
undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# Create alpha mask: 0 where pixels are pure black (from border), 255 elsewhere
gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
alpha = np.where(gray == 0, 0, 255).astype(np.uint8)

# Merge BGR + Alpha
bgra = cv2.cvtColor(undistorted, cv2.COLOR_BGR2BGRA)
bgra[:, :, 3] = alpha

out_path = "undistorted.png"
cv2.imwrite(out_path, bgra)

print(f"Saved with transparency: {out_path}")
