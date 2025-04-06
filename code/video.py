# Custom video writer
import tempfile
import subprocess
import torchvision
import shutil
import signal
import sys


class VideoWriter:
  def __init__(self, filename, fps, size, mbps=15):
    self.filename = filename
    self.fps = fps
    self.size = size
    self.mbps = mbps
    self.temp_dir = tempfile.mkdtemp(suffix="video_frames")
    # print(f"Temporary directory for video frames: {self.temp_dir}")
    self.frame_number = 0
    self.closed = False

    def cleanup_handler():
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
      shutil.rmtree(self.temp_dir)
      self.closed = True

  # Frame is a tensor of shape (H, W, C), BGRA
  def write_frame(self, frame):
    if self.closed:
      raise ValueError("VideoWriter has already been closed.")
    frame_updated = frame.permute(2, 0, 1) * (1.0 / 255.0)
    frame_rgba = frame_updated.clone()
    frame_rgba[0, :, :] = frame_updated[2, :, :]
    frame_rgba[2, :, :] = frame_updated[0, :, :]
    torchvision.utils.save_image(frame_rgba, f"{self.temp_dir}/frame_{self.frame_number:06d}.png")
    self.frame_number += 1

  def save_video(self):
    if self.closed:
      raise ValueError("VideoWriter has already been closed.")
    command = [
      "ffmpeg",
      "-y",
      "-framerate", str(self.fps),
      "-i", f"{self.temp_dir}/frame_%06d.png",
      "-c:v", "h264_videotoolbox",
      "-b:v", f"{self.mbps}M",
      "-pix_fmt", "yuv420p",
      self.filename
    ]
    try:
      subprocess.check_output(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print("FFmpeg error:", e.output.decode())
      raise

    self.__cleanup__()
