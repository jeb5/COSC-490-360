#!/bin/bash
python src/angle_estimation.py input/balloon_footage.MP4 input/gyro_data.csv temp/balloon_test.csv --output_video output/balloon_test.mp4 --start_frame 68000 --end_frame 70000 --debug