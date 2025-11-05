# Rotating 2D to Equirectangular 360 Video

Uses computer vision techniques to translate 2D video of a large scene, filmed with a rotating camera, into a stable 360 equirectangular video.

## Installation

```bash
git clone https://github.com/jeb5/COSC-490-360.git
cd COSC-490-360
pip install opencv-python torch torchvision pillow line-profiler av matplotlib scipy shapely progressbar2 git+https://github.com/google/spatial-media.git
```

## Usage

Create an input directory with the same format as `example_input/`. You need to provide a video file, a csv file with camera rotation data if available, and a `camera_information.json` file with camera parameters.

```bash
python src/VideoTo360.py example_input/ --produce_360
# See more options with:
python src/VideoTo360.py --help
```

## Visual Odometry
Uses SIFT feature detection and pairwise image matching to estimate relative camera orientation change between frames.
Global camera poses are then computed by combining all relative pose estimates and solving for an optimal set of camera orientations.

<img src="images/vo.svg" width="400" alt="Visual odometry process"/>

## Reprojection
Given an estimated camera orientation for each frame, 2D-to-equirectangular reprojection is performed, and the resulting equirectangular frames are stacked atop previous frames to create a full 360 video.

<img src="images/project_to_sphere.svg" width="600" alt="2D Video to Equirectangular format projection process"/>

## Example Result Screenshots
<img src="images/example_output1.png" width="800" alt="Example output frame 1"/>

<img src="images/rot_chain_vs_sliding.jpg" width="400" alt="Comparison of rotation chain vs sliding window optimization"/>
<img src="images/rot_fail_overlap_win.jpg" width="300" alt="Example of rotation chaing failure due to insufficient overlap"/>
