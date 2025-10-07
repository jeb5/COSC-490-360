# Custom video writer
import tempfile
import subprocess
import shutil
import helpers
import spatialmedia
import spatialmedia.metadata_utils
import threading
import atexit
import platform


class VideoWriter:
  def __init__(self, filename, fps, size, mbps=10, spherical_metadata=False):
    self.filename = filename
    self.fps = fps
    self.size = size
    self.mbps = mbps
    self.spherical_metadata = spherical_metadata
    self.temp_dir = tempfile.mkdtemp(suffix="video_writer_temp")
    extension = filename.split(".")[-1]
    self.temp_path = f"{self.temp_dir}/temp.{extension}"

    ffmpeg_command = [
      "ffmpeg",
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "bgr24",
      "-s",
      f"{size[0]}x{size[1]}",
      "-framerate",
      f"{fps:.4f}",
      "-i",
      "-",
      *(
        ["-c:v", "h264_videotoolbox"] if ((size[0] < 4096) and (platform.system() == "Darwin")) else []
      ),  # Use hardware encoding for smaller resolutions
      "-b:v",
      f"{self.mbps}M",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      self.temp_path,
    ]
    print(f"Starting ffmpeg with command: {' '.join(ffmpeg_command)}")

    self.ffmpeg_process = subprocess.Popen(
      ffmpeg_command,
      stdin=subprocess.PIPE,
      stderr=subprocess.PIPE,
      stdout=subprocess.PIPE,
    )

    def stream_output(stream, write_func):
      while True:
        line = stream.readline()
        if not line:
          break
        write_func(line.decode("utf-8"), end="")

    threading.Thread(target=stream_output, args=(self.ffmpeg_process.stdout, print), daemon=True).start()
    threading.Thread(target=stream_output, args=(self.ffmpeg_process.stderr, print), daemon=True).start()

    self.frame_number = 0
    self.closed = False
    self.dirty = False

    # If save_video is never called, cleanup the temp directory when the program exits
    atexit.register(self.__cleanup__)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.save_video()

  def __cleanup__(self):
    if not self.closed:
      shutil.rmtree(self.temp_dir)
      self.closed = True

  def did_write(self):
    return self.dirty

  # Frame is a tensor of shape (H, W, C), BGRA or BGR
  def write_frame(self, frame):
    if self.closed:
      raise ValueError("VideoWriter has already been closed.")
    frame = frame.clone()
    if frame.shape[-1] == 4:
      frame = helpers.BGRAToBGRAlphaBlack_torch(frame)
    frame = frame.byte().cpu().numpy()
    self.ffmpeg_process.stdin.write(frame.tobytes())
    self.frame_number += 1
    self.dirty = True

  def write_frame_opencv(self, frame):
    if self.closed:
      raise ValueError("VideoWriter has already been closed.")
    self.ffmpeg_process.stdin.write(frame.tobytes())
    self.frame_number += 1
    self.dirty = True

  def save_video(self):
    if self.closed:
      raise ValueError("VideoWriter has already been closed.")

    self.ffmpeg_process.stdin.close()
    self.ffmpeg_process.wait()

    if self.frame_number != 0:
      if self.spherical_metadata:
        addSphericalMetadata(self.temp_path, self.filename)
      else:
        shutil.move(self.temp_path, self.filename)
      print(f"Saved video to {self.filename} with {self.frame_number} frames.")

    self.__cleanup__()


def addSphericalMetadata(input_path, output_path):
  metadata = spatialmedia.metadata_utils.Metadata()
  metadata.video = spatialmedia.metadata_utils.generate_spherical_xml()

  def logging(message):
    print(message)

  spatialmedia.metadata_utils.inject_metadata(input_path, output_path, metadata, logging)
