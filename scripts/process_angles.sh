#!/bin/bash
python src/angle_estimation.py input/balloon_footage.MP4 input/gyro_data.csv output/balloon_sample.mp4 output/balloon_sample.csv 
--start_frame 68000 --end_frame 78000