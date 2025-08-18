import sys
from scipy.spatial.transform import Rotation as R
import numpy as np


def read_file(path):
  data = {}
  with open(path, "r") as f:
    for line in f:
      parts = line.strip().split(",")
      if len(parts) < 5:
        continue
      i, j = int(parts[0]), int(parts[1])
      euler = list(map(float, parts[2:5]))
      # assuming XYZ intrinsic convention, input already in degrees
      data[(i, j)] = R.from_euler("xyz", euler, degrees=True)
  return data


def main(file1, file2, outfile):
  data1 = read_file(file1)
  data2 = read_file(file2)

  common = data1.keys() & data2.keys()
  only1 = data1.keys() - data2.keys()
  only2 = data2.keys() - data1.keys()

  with open(outfile, "w") as out:
    # Write angle differences
    for key in sorted(common):
      r1 = data1[key]
      r2 = data2[key]
      # relative rotation
      rel = r1.inv() * r2
      angle_rad = rel.magnitude()
      angle_deg = np.degrees(angle_rad)
      out.write(f"{key[0]},{key[1]},{angle_deg:.6f}\n")

    # Report missing pairs
    out.write("\n# Pairs only in first file:\n")
    for key in sorted(only1):
      out.write(f"{key[0]},{key[1]}\n")

    out.write("\n# Pairs only in second file:\n")
    for key in sorted(only2):
      out.write(f"{key[0]},{key[1]}\n")


if __name__ == "__main__":
  if len(sys.argv) != 4:
    print("Usage: python compare_rotations.py file1 file2 output")
    sys.exit(1)
  main(sys.argv[1], sys.argv[2], sys.argv[3])
