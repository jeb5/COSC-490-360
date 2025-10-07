import math
import numpy as np
from scipy.spatial import ckdtree
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class KeypointsManager:
  def __init__(self, K):
    self.kpLocs = generate_fibonacci_sphere_points(K)
    self.tree = ckdtree.cKDTree(self.kpLocs)
    self.keypoints = [None] * K

  def add_potential_keypoint(self, rotation, features, index):
    rot_matrix = rotation.as_matrix()
    position = rot_matrix @ np.array([0, 1, 0])
    nearest_dist, nearest_idx = self.tree.query(position)
    self.keypoints[nearest_idx] = {
      "rotation": rotation,
      "features": features,
      "index": index,
    }

  def get_closest_keypoint(self, rotation):
    rot_matrix = rotation.as_matrix()
    position = rot_matrix @ np.array([0, 1, 0])
    nearest_dist, nearest_idx = self.tree.query(position)
    return self.keypoints[nearest_idx]

  def visualize_keypoints3(self, dm, img_scale=0.12, img_res=32):
    """
    Visualize keypoints in 3D with true camera-facing billboards.

    Args:
        self: object with `kpLocs` (N x 3 list/array)
        keypoints: list where each element is None or an object with 'index'
        dm: object providing get_frame(index) -> numpy array image (H,W) or (H,W,3) or (H,W,4)
        img_scale: size of billboard in data units (roughly the height)
        img_res: texture resolution (height in pixels used to map the image onto the surface).
                 Lower for speed (e.g., 24 or 32), higher for visual quality.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Enforce square aspect ratio (Matplotlib >=3.3). If not available, try best-effort.
    try:
      ax.set_box_aspect((1, 1, 1))
    except Exception:
      pass

    # Hide axes, ticks, background
    ax.set_axis_off()

    # Collect points and billboards
    billboard_specs = []
    scatter_pts = []

    xs = []
    ys = []
    zs = []

    for idx, kp in enumerate(self.keypoints):
      x, y, z = tuple(self.kpLocs[idx])
      xs.append(x)
      ys.append(y)
      zs.append(z)

      if kp is None:
        scatter_pts.append((x, y, z))
      else:
        img = np.asarray(dm.get_frame(kp["index"]))
        # convert to float0-1
        imgf = img.astype(np.float32)
        if imgf.max() > 1.0:
          imgf = imgf / 255.0
        # Convert BGR to RGB
        imgf = imgf[..., ::-1]

        billboard_specs.append({"pos": np.array([x, y, z], dtype=float), "img": imgf})

    # Plot black dots for None keypoints
    if scatter_pts:
      px, py, pz = zip(*scatter_pts)
      ax.scatter(px, py, pz, c="k", s=20)

    # set sane limits & padding (so camera computations are stable)
    if len(xs) == 0:
      return
    allx = np.array(xs)
    ally = np.array(ys)
    allz = np.array(zs)
    xmin, xmax = allx.min(), allx.max()
    ymin, ymax = ally.min(), ally.max()
    zmin, zmax = allz.min(), allz.max()

    # compute center and a useful radius
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    xrange = xmax - xmin
    yrange = ymax - ymin
    zrange = zmax - zmin
    max_range = max(xrange, yrange, zrange, 1e-6)
    pad = max_range * 0.15
    half = max_range / 2.0 + pad

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

    # nearest-neighbour resize (fast, no external deps)
    def resize_nn(img, H, W):
      src_h, src_w = img.shape[:2]
      row_idx = (np.linspace(0, src_h - 1, H)).astype(int)
      col_idx = (np.linspace(0, src_w - 1, W)).astype(int)
      return img[np.ix_(row_idx, col_idx)]

    surfaces = []  # store created surface objects so we can remove them

    def draw_billboards(event=None):
      # remove old surfaces (only the ones we created)
      while surfaces:
        s = surfaces.pop()
        try:
          s.remove()
        except Exception:
          pass

      # camera position estimation (spherical coords around axes center)
      azim = np.deg2rad(ax.azim)
      elev = np.deg2rad(ax.elev)
      # unit direction corresponding to azim/elev
      cam_dir_unit = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)], dtype=float)
      # place camera a distance proportional to axis size from the axes center
      cam_distance = max_range * 3.0
      camera_pos = np.array([cx, cy, cz]) + cam_dir_unit * cam_distance

      for spec in billboard_specs:
        center = spec["pos"]
        imgf = spec["img"]
        h0, w0 = imgf.shape[:2]
        aspect = float(w0) / float(h0)

        # target resolution for texture mapping
        H = max(2, int(img_res))
        W = max(2, int(round(img_res * aspect)))
        tex = resize_nn(imgf, H, W)

        # vector from billboard to camera
        cam_vec = camera_pos - center
        nrm = np.linalg.norm(cam_vec)
        if nrm < 1e-9:
          continue
        cam_vec = cam_vec / nrm  # normal pointing toward camera

        # choose 'right' and 'up' for billboard plane
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(cam_vec, world_up)
        if np.linalg.norm(right) < 1e-6:
          # camera aligned with world_up; pick a different up
          world_up = np.array([0.0, 1.0, 0.0])
          right = np.cross(cam_vec, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, cam_vec)
        up = up / np.linalg.norm(up)

        # extents (height controlled by img_scale)
        half_h = img_scale / 2.0
        half_w = (img_scale * aspect) / 2.0

        u = np.linspace(-half_w, half_w, W)
        v = np.linspace(-half_h, half_h, H)
        U, V = np.meshgrid(u, v)  # shape (H, W)
        # build the 3D coordinates of the billboard mesh
        X = center[0] + right[0] * U + up[0] * V
        Y = center[1] + right[1] * U + up[1] * V
        Z = center[2] + right[2] * U + up[2] * V

        # prepare facecolors: shape (H, W, 4)
        fc = np.array(tex, dtype=float)
        if fc.max() > 1.0:
          fc = fc / 255.0
        if fc.shape[2] == 3:
          alpha = np.ones((H, W, 1), dtype=float)
          fc = np.concatenate([fc, alpha], axis=2)
        # ensure values clipped to [0,1]
        fc = np.clip(fc, 0.0, 1.0)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fc, shade=False)
        surfaces.append(surf)

      # redraw
      fig.canvas.draw_idle()

    # hook events that change view
    fig.canvas.mpl_connect("draw_event", draw_billboards)
    fig.canvas.mpl_connect("button_release_event", draw_billboards)
    fig.canvas.mpl_connect("motion_notify_event", draw_billboards)

    # initial draw
    draw_billboards()
    plt.show()


def generate_fibonacci_sphere_points(n):
  points = []
  phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

  for i in range(n):
    y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
    radius = math.sqrt(1 - y * y)  # radius at y

    theta = phi * i  # golden angle increment

    x = math.cos(theta) * radius
    z = math.sin(theta) * radius

    # rot = R.align_vectors([[0, 1, 0]], [[x, y, z]])[0]
    # points.append((np.array([x, y, z]), rot))

    points.append([x, y, z])

  return np.asarray(points)

  # self.keypoints = {rotation:np.array, index: int}[]
