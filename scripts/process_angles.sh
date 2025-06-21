#!/bin/bash
# python src/angle_estimation.py input/balloon_footage.MP4 input/gyro_data.csv temp/balloon_test.csv --output_video output/balloon_test.mp4 --start_frame 68000 --end_frame 70000 --gopro --output_debug_video output/balloon_test_debug.mp4
python src/angle_estimation.py input/synthetic2.mkv input/synthetic2.csv temp/test.csv --output_debug_video output/test.mp4
# python src/angle_estimation.py input/synthetic3.mkv input/synthetic3.csv temp/test.csv --output_debug_video output/test.mp4