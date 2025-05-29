# Custom video writer
import tempfile
import subprocess
import torch
import torchvision
import shutil
import signal
import sys
import helpers
import spatialmedia
import av
import spatialmedia.metadata_utils
import line_profiler


class VideoWriter:
  def __init__(self, filename, fps, size, mbps=15, spherical_metadata=False):
    self.filename = filename
    self.fps = fps
    self.size = size
    self.mbps = mbps
    self.spherical_metadata = spherical_metadata
    self.temp_dir = tempfile.mkdtemp(suffix="video_writer_temp")
    extension = filename.split(".")[-1]
    self.temp_path = f"{self.temp_dir}/temp.{extension}"
    self.container = av.open(self.temp_path, 'w')
    self.stream = self.container.add_stream('h264', rate=fps)
    self.stream.width = size[0]
    self.stream.height = size[1]
    self.stream.pix_fmt = 'yuv420p'
    self.stream.bit_rate = mbps * 1000000

    self.frame_number = 0
    self.closed = False

    def cleanup_handler(signum, frame):
      self.__cleanup__()
      sys.exit(0)
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.save_video()

  def __cleanup__(self):
    if not self.closed:
      # shutil.rmtree(self.temp_dir)
      self.closed = True

  # Frame is a tensor of shape (H, W, C), BGRA
  @line_profiler.profile
  def write_frame(self, frame):
    if self.closed:
      raise ValueError("VideoWriter has already been closed.")
    frame_updated = helpers.BGRAToBGRAlphaBlack_torch(frame)
    frame_updated = frame_updated.byte().cpu().numpy()
    frame_updated = av.VideoFrame.from_ndarray(frame_updated, format='bgr24')
    for packet in self.stream.encode(frame_updated):
      self.container.mux(packet)
    self.frame_number += 1

  @line_profiler.profile
  def write_frame_opencv(self, frame):
    if self.closed:
      raise ValueError("VideoWriter has already been closed.")
    frame_updated = av.VideoFrame.from_ndarray(frame, format='bgr24')
    for packet in self.stream.encode(frame_updated):
      self.container.mux(packet)
    self.frame_number += 1

  def save_video(self):
    if self.closed:
      raise ValueError("VideoWriter has already been closed.")

    for packet in self.stream.encode():
      self.container.mux(packet)
    self.container.close()

    # output_path = self.filename
    # if self.spherical_metadata:
    #   video_extension = self.filename.split(".")[-1]
    #   output_path = self.temp_dir + "/output_temp" + "." + video_extension

    # command = [
    #   "ffmpeg",
    #   "-y",
    #   "-framerate", str(self.fps),
    #   "-i", f"{self.temp_dir}/frame_%06d.png",
    #   "-c:v", "h264_videotoolbox",
    #   "-b:v", f"{self.mbps}M",
    #   "-pix_fmt", "yuv420p",
    #   output_path
    # ]
    # try:
    #   subprocess.check_output(command, stderr=subprocess.STDOUT)
    # except subprocess.CalledProcessError as e:
    #   print("FFmpeg error:", e.output.decode())
    #   self.__cleanup__()
    #   raise
    if self.frame_number != 0:
      if self.spherical_metadata:
        addSphericalMetadata(self.temp_path, self.filename)
      else:
        shutil.move(self.temp_path, self.filename)

    self.__cleanup__()


def addSphericalMetadata(input_path, output_path):
  metadata = spatialmedia.metadata_utils.Metadata()
  metadata.video = spatialmedia.metadata_utils.generate_spherical_xml()
  def logging(message): print(message)
  spatialmedia.metadata_utils.inject_metadata(input_path, output_path, metadata, logging)
