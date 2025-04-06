import cv2 as cv
import remap
import helpers


def main():
  # Load the video
  video_path = "./input/bsrc8.mkv"
  video = cv.VideoCapture(video_path)
  if not video.isOpened():
    print("Error: Could not open video.")
    return


if __name__ == "__main__":
  main()
