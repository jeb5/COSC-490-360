import csv
import numpy as np
import cv2 as cv
import PIL.ImageDraw
import PIL.ImageFont
import PIL.Image
import argparse
from importlib.resources import files


def main(args):

  number_of_lines = -1  # Skip header
  with open(args.gyro_csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      number_of_lines += 1
  gyro = np.zeros((number_of_lines, 3)).astype(np.float32)
  pitch_yaw_roll = np.zeros((number_of_lines, 3)).astype(np.float32)
  acc = np.zeros((number_of_lines, 3)).astype(np.float32)
  frame_number = np.zeros((number_of_lines, 1)).astype(np.uint32)
  with open(args.gyro_csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for i, row in enumerate(reader):
      frame_number[i] = [float(row[0])]
      # XYZ
      gyro[i] = [float(row[2]), float(row[3]), float(row[4])]
      # Pitch, Yaw, Roll
      pitch_yaw_roll[i] = [float(row[5]), float(row[6]), float(row[7])]
      # XYZ
      acc[i] = [float(row[8]), float(row[9]), float(row[10])]

  input_video = cv.VideoCapture(args.video_path)
  if not input_video.isOpened():
    print("Error opening video file")
    return
  size = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  last_frame = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))
  input_framerate = int(input_video.get(cv.CAP_PROP_FPS))
  output_video = cv.VideoWriter(args.output_video_path, cv.VideoWriter_fourcc(*'mp4v'), input_framerate, size)
  i = 0
  (start_frame, end_frame) = (args.start_frame, args.end_frame if args.end_frame >= 0 else last_frame)
  with open(args.output_angle_path, 'w') as csvfile:
    for frame in range(start_frame, end_frame, 1):
      indices = np.where(frame_number == frame)[0]
      pitch_yaw_rolls = pitch_yaw_roll[indices]
      # average_pitch_yaw_roll = np.mean(pitch_yaw_rolls, axis=0)
      pitch_yaw_roll_frame = pitch_yaw_rolls[0]

      pitch = pitch_yaw_roll_frame[0]
      yaw = pitch_yaw_roll_frame[1]
      roll = pitch_yaw_roll_frame[2]
      yaw, roll = roll, yaw  # These seem to be swapped in the CSV file
      pitch = pitch - 90
      roll = -roll
      # Get video frame
      input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
      ret, image = input_video.read()
      if not ret:
        print("Error reading video frame")
        break
      if args.add_angle_text:
        angleText = "Pitch: {:.1f}\nYaw: {:.1f}\nRoll: {:.1f}".format(
          pitch, yaw, roll)
        image = add_text_to_image(image, angleText)
      output_video.write(image)
      print(f"\rWriting frame {i}/{end_frame-start_frame}", end="")
      toRad = np.pi / 180
      csvfile.write("{},{},{},{}\n".format(i, pitch * toRad, yaw * toRad, roll * toRad))
      i += 1
  print()

  input_video.release()
  output_video.release()


def add_text_to_image(image, text):
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  pil_image = PIL.Image.fromarray(image)
  draw = PIL.ImageDraw.Draw(pil_image)
  with files("src").open("roboto.ttf") as font_file:
    font = PIL.ImageFont.truetype(font_file, 30)
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
  image_cv = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
  return image_cv


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Estimate angles from gyro data and video.")
  parser.add_argument("video_path", type=str, help="Path to the input video file.")
  parser.add_argument("gyro_csv_path", type=str, help="Path to the gyro CSV file.")
  parser.add_argument("output_video_path", type=str, help="Path to the output video file.")
  parser.add_argument("output_angle_path", type=str, help="Path to the output angle CSV file.")
  parser.add_argument("--start_frame", type=int, default=0, help="Start frame for processing.")
  parser.add_argument("--end_frame", type=int, default=-1, help="End frame for processing.")
  parser.add_argument("--add_angle_text", action="store_true", help="Add angle text to video frames.")
  args = parser.parse_args()
  main(args)
