import math
import numpy as np
import cv2 as cv
import torch
import torchvision
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import progressbar as pb

font_path = 'src/assets/roboto_mono.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.monospace'] = fm.FontProperties(fname=font_path).get_name()

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

  return outImage.astype(np.uint8)

# Images are BGRA [h, w, 4]


def paste_cv(background, foreground, x, y):
  fgx1, fgy1 = max(-x, 0), max(-y, 0)
  fgx2, fgy2 = min(x + foreground.shape[1], background.shape[1]) - x, min(y + foreground.shape[0], background.shape[0]) - y
  if fgx1 >= fgx2 or fgy1 >= fgy2:
    return background
  bgx1, bgy1 = x + fgx1, y + fgy1
  bgx2, bgy2 = x + fgx2, y + fgy2
  background[bgy1:bgy2, bgx1:bgx2] = foreground[fgy1:fgy2, fgx1:fgx2]
  return background

def add_transparent_image_torch(background, foreground):
  fg_alpha = foreground[:, :, 3] / 255.0
  bg_alpha = background[:, :, 3] / 255.0

  fg_color = foreground[:, :, :3] / 255.0
  bg_color = background[:, :, :3] / 255.0

  output_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
  fg_alpha_3 = torch.stack((fg_alpha, fg_alpha, fg_alpha), dim=-1)
  fg_color = fg_color * fg_alpha_3
  bg_color = bg_color * (1 - fg_alpha_3)

  outImage = fg_color + bg_color
  outImage = torch.cat((outImage, output_alpha.unsqueeze(-1)), dim=-1)
  outImage = outImage * 255.0
  return outImage


def display_torch_image(image, window_name="Image"):
  image_cv = BGRAToBGRAlphaBlack_torch(image).cpu().numpy().astype(np.uint8)
  cv.imshow(window_name, image_cv)
  cv.waitKey(0)
  cv.destroyAllWindows()

def centerCrop(image, target_width, target_height):
  top, left = (image.shape[0] - target_height) // 2, (image.shape[1] - target_width) // 2
  cropped = image.permute(2, 0, 1)
  cropped = torchvision.transforms.functional.crop(cropped, top, left, target_height, target_width)
  cropped = cropped.permute(1, 2, 0)
  return cropped


def BGRAToBGRAlphaBlack(image):
  if image.shape[2] != 4:
    raise ValueError("Image must have 4 channels")
  if image.dtype != np.uint8:
    raise ValueError("Image must be of type uint8")

  bgr = image[:, :, :3]
  alpha = image[:, :, 3]
  alpha = alpha.astype(float) / 255.0
  alpha = cv.merge((alpha, alpha, alpha))
  bgr = cv.multiply(bgr.astype(float), alpha)
  return bgr.astype(np.uint8)


# Image is BGRA [h, w, 4]


def BGRAToBGRAlphaBlack_torch(image):
  if image.shape[2] != 4:
    raise ValueError("Image must have 4 channels")
  if image.dtype != torch.float32:
    raise ValueError("Image must be of type float32")

  bgr = image[:, :, :3]
  alpha = image[:, :, 3]
  alpha = alpha / 255.0
  alpha = torch.stack((alpha, alpha, alpha), dim=-1)
  bgr = bgr * alpha
  return bgr


def get_device():
  device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
  if device == "cpu":
    print("Warning: Using CPU for remapping, which may be slow")
  return device


def apply_combined_vignette_alpha(image, circ_start_pct, circ_end_pct=None, rect_start_pct=0.0, rect_end_pct=None):
  """
  Applies a combined circular and rectangular vignette alpha to an (H, W, 4) image tensor.
  """
  assert image.ndim == 3 and image.shape[2] == 4
  H, W, _ = image.shape
  device = image.device

  # Create coordinate grid once
  y = torch.linspace(0, H - 1, H, device=device)
  x = torch.linspace(0, W - 1, W, device=device)
  yy, xx = torch.meshgrid(y, x, indexing="ij")

  # Precompute common terms
  cx, cy = W / 2, H / 2
  dx = xx - cx
  dy = yy - cy
  dist_from_center = torch.sqrt(dx * dx + dy * dy)

  # Max radius is half-diagonal length
  max_radius = (H**2 + W**2) ** 0.5 / 2
  circ_start = circ_start_pct * max_radius
  circ_end = circ_end_pct * max_radius if circ_end_pct is not None else circ_start

  # Circular alpha
  if circ_end == circ_start:
    circ_alpha = (dist_from_center <= circ_start).float()
  else:
    circ_alpha = torch.clamp((circ_end - dist_from_center) / (circ_end - circ_start), 0.0, 1.0)

  # Rectangular distance to nearest edge
  dist_to_edge = torch.minimum(torch.minimum(yy, H - 1 - yy), torch.minimum(xx, W - 1 - xx))

  # Rect alpha
  min_dim = min(H, W)
  rect_start = rect_start_pct * min_dim
  rect_end = rect_end_pct * min_dim if rect_end_pct is not None else rect_start

  if rect_end == rect_start:
    rect_alpha = (dist_to_edge >= rect_start).float()
  else:
    rect_alpha = torch.clamp((dist_to_edge - rect_start) / (rect_end - rect_start), 0.0, 1.0)

  # Final alpha mask = circular * rectangular
  alpha_mask = circ_alpha * rect_alpha

  # Apply to alpha channel
  result = image.clone()
  result[..., 3].mul_(alpha_mask)

  return result

def get_sequence(n, first_consecutive, max):
  """
  Generate an increasing sequence of n numbers. The first `first_consecutive` numbers are consecutive integers starting from 1. The rest are generated with a cubic function, with the n-1th number being `max`.
  """

  def f(x, a=10, b=30, c=3000):
    if x < a:
      return x
    else:
      x = x - a
      b = b - a
      # cubic function with f'(a) = 1, f(a) = a, f(b) = max
      return math.floor(((c - a - b) / math.pow(b, 3)) * math.pow(x, 3) + x + a)

  sequence = [f(i + 1, first_consecutive, n, max) for i in range(n)]
  return sequence


class ProcessContext:
  def __init__(self, max_value, prefix_widgets, swallow_keyboard_interrupts=True):
    self.bar = pb.ProgressBar(
      max_value=max_value,
      widgets=prefix_widgets + [" ", pb.GranularBar(), " ", pb.ETA()],
      redirect_stdout=True,
      redirect_stderr=True,
    )
    self.swallow_keyboard_interrupts = swallow_keyboard_interrupts

  def __enter__(self):
    return self.bar

  def __exit__(self, exc_type, exc_value, traceback):
    self.bar.finish(dirty=True)
    print()
    if exc_type is KeyboardInterrupt and self.swallow_keyboard_interrupts:
      print("KeyboardInterrupt. Cancelling...")
      return True  # Suppress the exception
    return False


def generate_rotation_histories_plot(rotation_histories, extra_text=None, extra_rot=None, interactive=False, legend=True):
  text_fig = None
  if extra_text is not None:
    text_fig = plt.figure(figsize=(3, 1.4))
    ax_text = text_fig.add_subplot()
    ax_text.axis("off")
    ax_text.text(
      0,
      1.0,
      extra_text,
      fontsize=12,
      ha="left",
      va="top",
      wrap=True,
      color="black",
      transform=ax_text.transAxes,
      fontfamily="monospace",
      fontweight="bold",
    )

    text_fig.tight_layout()

  # Main 3D plot
  main_fig = plt.figure(figsize=(3, 3))
  ax_main = main_fig.add_subplot(projection="3d")
  ax_main.set_box_aspect(aspect=None, zoom=1)
  for rotation_history in rotation_histories:
    name = rotation_history["name"]
    colour = rotation_history["colour"]
    rotations = rotation_history["data"]
    # Get mask to filter out None values
    mask = [rot is not None for rot in rotations]
    mask = np.array(mask)
    rotations = [rot if rot is not None else np.eye(3) for rot in rotations]
    rotations = np.array(rotations)
    # Python version (old)
    # vectors = [rotation @ np.array([0, 1, 0]) for rotation in rotations]
    # Numpy version (new)
    vectors = rotations @ np.array([0, 1, 0])
    # Apply mask, setting None values to NaN
    vectors = np.where(mask[:, None], vectors, np.nan)

    vector_history = np.array(vectors)
    xs, ys, zs = vector_history[:, 0], vector_history[:, 1], vector_history[:, 2]
    ax_main.quiver(
      0,
      0,
      0,
      xs[-1],
      ys[-1],
      zs[-1],
      length=1,
      normalize=True,
      color=colour,
      arrow_length_ratio=0.2,
    )
    ax_main.plot(xs, ys, zs, c=colour, marker=".", label=name)

  if extra_rot is not None:
    extra_vector = extra_rot @ np.array([0, 1, 0])
    ax_main.quiver(
      0,
      0,
      0,
      extra_vector[0],
      extra_vector[1],
      extra_vector[2],
      length=1,
      normalize=True,
      color="red",
      arrow_length_ratio=0.2,
    )

  ax_main.set_xlabel("X", labelpad=-10)
  ax_main.set_ylabel("Y", labelpad=-10)
  ax_main.set_zlabel("Z", labelpad=-10)
  ax_main.set_xlim(-1, 1)
  ax_main.set_ylim(-1, 1)
  ax_main.set_zlim(-1, 1)
  if legend:
    ax_main.legend()
  ax_main.set_xticklabels([])
  ax_main.set_yticklabels([])
  ax_main.set_zticklabels([])

  if interactive:
    main_fig.show()
    plt.show(block=True)

  main_image = figure_to_cv_image(main_fig)
  if text_fig is None:
    return main_image
  text_image = figure_to_cv_image(text_fig)
  return np.vstack((text_image, main_image))


def figure_to_cv_image(fig):
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height(physical=True)
  plt.close(fig)
  image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
  image = image.reshape(h, w, 4)
  image = image[:, :, 1:4]
  image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

  return image