import csv
import math
import os
import pickle
import numpy as np
import cv2 as cv
import PIL.ImageDraw
import PIL.ImageFont
import PIL.Image
import matplotlib.pyplot as plt
import argparse
import helpers
import torch
import sys
import signal
from video_writer import VideoWriter
from scipy.spatial.transform import Rotation as R


def main(args):

    input_video = cv.VideoCapture(args.video_path)
    if not input_video.isOpened():
        print("Error opening video file")
        return

    input_size = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        input_video.get(cv.CAP_PROP_FRAME_HEIGHT)
    )
    input_framerate = int(input_video.get(cv.CAP_PROP_FPS))
    (start_frame, end_frame) = (
        args.start_frame,
        (
            args.end_frame
            if args.end_frame >= 0
            else int(input_video.get(cv.CAP_PROP_FRAME_COUNT) - 1)
        ),
    )

    cam_matrix, cam_distortion = (
        helpers.GOPRO_CAMERA if args.gopro else helpers.BLENDER_CAMERA
    )
    new_cam_mat = (
        cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            cam_matrix, cam_distortion, input_size, None, None, 1, input_size, 1
        )
        if args.gopro
        else cam_matrix
    )
    if args.gopro:
        m1, m2 = cv.fisheye.initUndistortRectifyMap(
            cam_matrix, cam_distortion, None, new_cam_mat, input_size, cv.CV_32FC1
        )

    if args.output_video is not None:
        output_video = VideoWriter(args.output_video, input_framerate, input_size)
    if args.output_debug_video is not None:
        output_debug_video = VideoWriter(
            args.output_debug_video, input_framerate, input_size
        )
        # IF output_debug_video was disabled, we *could* save time by not undistorting the image, only the feature locations

    def cleanup():
        input_video.release()
        if args.output_video is not None:
            output_video.save_video()
        if args.output_debug_video is not None:
            output_debug_video.save_video()
        print("Done.")

    def interupt_handler(signum, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, interupt_handler)

    visual_rotations = []
    inertial_rotations = []
    frame_features = []

    # def serialize_keypoints(kp):
    #     return [
    #         (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
    #         for kp in kp
    #     ]

    # def deserialize_keypoints(serialized_kp):
    #     return [
    #         cv.KeyPoint(x, y, size, angle, response, octave, class_id)
    #         for (x, y, size, angle, response, octave, class_id) in serialized_kp
    #     ]

    # # Load serialized f"{video_path}/sift.data" into frame_features if it exists
    # if os.path.exists(f"{args.video_path}.sift.data"):
    #     with open(f"{args.video_path}.sift.data", "rb") as f:
    #         frame_features_serialized = pickle.load(f)
    #         frame_features = [
    #             (deserialize_keypoints(kp), np.asarray(des, dtype=np.float32))
    #             for (kp, des) in frame_features_serialized
    #         ]
    #     print(f"Loaded {len(frame_features)} frames from sift.data")

    # if len(frame_features) == 0:
    #     with open(args.gyro_csv_path, "r") as csvfile:
    #         for frame, xyz in inertials_from_csv(args, start_frame, end_frame):
    #             input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
    #             ret, image = input_video.read()
    #             # print("Read frame ", frame)
    #             image_undistorted = (
    #                 cv.remap(image, m1, m2, cv.INTER_LINEAR) if args.gopro else image
    #             )
    #             # print("Undistorted frame ", frame)
    #             frame_features.append(get_features(image_undistorted))
    #             print("Processed frame", frame)
    #     with open(f"{args.video_path}.sift.data", "wb") as f:

    #         serialized_frame_features = [
    #             (serialize_keypoints(kp), des.tolist()) for (kp, des) in frame_features
    #         ]
    #         pickle.dump(serialized_frame_features, f)

    # # im1, im2 = 87, 37
    # # input_video.set(cv.CAP_PROP_POS_FRAMES, im1)
    # # frame1 = input_video.read()[1]
    # # # sift1 = get_features(frame)
    # # sift1 = frame_features[im1]
    # # input_video.set(cv.CAP_PROP_POS_FRAMES, im2)
    # # frame2 = input_video.read()[1]
    # # # sift2 = get_features(frame2)
    # # sift2 = frame_features[im2]
    # # match_image = get_angle_difference(sift1, sift2, new_cam_mat, frame1, frame2)[1]
    # # # Display the match image
    # # cv.imshow("Match Image", match_image)
    # # cv.waitKey(0)
    #
    # solve_rotations(frame_features, new_cam_mat)
    # cleanup()

    with open(args.output_angle_path, "w") as csvfile:
        for frame, xyz in inertials_from_csv(args, start_frame, end_frame):

            input_video.set(cv.CAP_PROP_POS_FRAMES, frame)
            ret, image = input_video.read()
            if not ret:
                print("Error reading video frame")
                break
            image_undistorted = (
                cv.remap(image, m1, m2, cv.INTER_LINEAR) if args.gopro else image
            )
            frame_features.append(get_features(image_undistorted))

            xyz = np.array(xyz)
            zxy = np.array([xyz[2], xyz[0], xyz[1]])

            intertial_rotation = R.from_euler("ZXY", zxy, degrees=True).as_matrix()
            inertial_rotations.append(intertial_rotation)
            visual_rotation_change, match_image = None, None
            if frame == start_frame:
                visual_rotations.append(inertial_rotations[-1])
                match_image = image_undistorted.copy()
            else:
                # for back in range(min(frame - start_frame, 2), 0, -1):
                back = 1
                visual_rotation_change, match_image = get_angle_difference(
                    frame_features[-1 - back],
                    frame_features[-1],
                    new_cam_mat,
                    image_undistorted,
                )
                if visual_rotation_change is None:
                    raise Exception(
                        "Not enough good points found for homography estimation."
                    )
                else:
                    visual_rotations.append(
                        visual_rotations[-back] @ visual_rotation_change
                    )

            print(
                f"Writing frame {frame - start_frame + 1}/{end_frame - start_frame + 1}"
            )

            visual_rot_ZXY = R.from_matrix(visual_rotations[-1]).as_euler(
                "ZXY", degrees=True
            )
            if args.output_debug_video is not None:
                angleText = ""
                angleText += (
                    f"Frame {frame - start_frame + 1}/{end_frame - start_frame + 1}"
                )
                angleText += (
                    f"\nInertial:\n{zxy[0]:6.2f}y {zxy[1]:6.2f}p {zxy[2]:6.2f}r"
                )
                angleText += f"\nVisual:\n{visual_rot_ZXY[0]:6.2f}y {visual_rot_ZXY[1]:6.2f}p {visual_rot_ZXY[2]:6.2f}r"
                if visual_rotation_change is not None:
                    visual_rot_ZXY_change = R.from_matrix(
                        visual_rotation_change
                    ).as_euler("ZXY", degrees=True)
                    angleText += f"\nChange:\n{visual_rot_ZXY_change[0]:6.2f}y {visual_rot_ZXY_change[1]:6.2f}p {visual_rot_ZXY_change[2]:6.2f}r"

                vector_plot = generate_rotation_histories_plot(
                    [
                        {
                            "name": "Visual",
                            "colour": "#42a7f5",
                            "data": visual_rotations,
                        },
                        {
                            "name": "Inertial",
                            "colour": "#f07d0a",
                            "data": inertial_rotations,
                        },
                    ],
                    extra_text=angleText,
                    # extra_rot=visual_rotation_change,
                )
                (x, y) = (
                    match_image.shape[1] - vector_plot.shape[1],
                    match_image.shape[0] - vector_plot.shape[0],
                )
                match_image = helpers.paste_cv(match_image, vector_plot, x, y)

                # image_debug = add_text_to_image(match_image, angleText)
                output_debug_video.write_frame_opencv(match_image)
            if args.output_video is not None:
                output_video.write_frame_opencv(image)

            # Outputtting visual rotation (in xyz order)
            csvfile.write(
                "{},{},{},{}\n".format(
                    frame - start_frame,
                    visual_rot_ZXY[1],
                    visual_rot_ZXY[2],
                    visual_rot_ZXY[0],
                )
            )
    cleanup()


def add_text_to_image(image, text):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(pil_image)
    font = PIL.ImageFont.truetype("src/assets/roboto.ttf", 30)
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    image_cv = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
    return image_cv


def solve_rotations(sift_features, cameraMatrix):
    # Take list of frame features (SIFT lists)
    # For each frame, get with ALL previous frames (n^2, or more precisely, n(n-1)/2)
    # For each match pair, estimate rotation using homography. If not enough points, skip.
    # We will make a note of any frames that had no matches. (OR frames with no connection to the first frame. Union-Find?)
    # We then construct a matrix A using only the frames that are inter-matched.
    # To solve Az = 0, we construct A^TA, and take the 3 eigenvectors corresponding to the 3 smallest eigenvalues. This is equivilent to taking the 3 right singular vectors corresponding to the 3 smallest singular values of A
    # Now given 3 values of z, we reconstruct the rotation matrices R^{i} for each frame i.
    # These will be rotation matrices in a common coordinate frame, however not the world coordinate frame.
    # If we know the world coordinate frame of the first frame, we can rotate all matricies to the world coordinate frame.
    matches = []
    for i in range(len(sift_features)):
        for j in range(0, i):
            match_rot, _ = get_angle_difference(
                sift_features[i], sift_features[j], cameraMatrix
            )
            if match_rot is not None:
                matches.append((i, j, match_rot))

                print(f"Match found between frames {i} and {j}")

    pass


def get_features(frame):
    kp, des = get_features.sift.detectAndCompute(frame, None)
    # kp = get_features.orb.detect(frame, None)
    # kp, des = get_features.orb.compute(frame, kp)
    return (kp, des)


get_features.sift = cv.SIFT_create()
# get_features.orb = cv.ORB_create()


def inertials_from_csv(args, first_frame, last_frame):
    csv_path = args.gyro_csv_path
    gopro = args.gopro
    current_frame = first_frame
    frame_with_xyz = None
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        next(reader)
        for i, row in enumerate(reader):
            frame_number = int(row[0])
            if frame_number < first_frame:
                continue
            if frame_number > last_frame:
                # yield frame_with_xyz
                break
            if frame_number > current_frame:
                current_frame = frame_number
                yield frame_with_xyz
            frame_with_xyz = (
                (current_frame, (float(row[5]) - 90, float(row[6]), float(row[7])))
                if gopro
                else (current_frame, (float(row[1]), float(row[2]), float(row[3])))
            )
        if current_frame == last_frame:
            yield frame_with_xyz


def get_angle_difference(features1, features2, cameraMatrix, frame1=None, frame2=None):
    kp1, des1 = features1
    kp2, des2 = features2

    matches = get_angle_difference.flann.knnMatch(des1, des2, k=2)

    # # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    good_points_1 = []
    good_points_2 = []
    for i, matchPairs in enumerate(matches):
        if len(matchPairs) != 2:
            print(len(matchPairs))
            continue
        m, n = matchPairs
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good_points_1.append(kp1[m.queryIdx].pt)
            good_points_2.append(kp2[m.trainIdx].pt)
    good_points_1 = np.array(good_points_1)
    good_points_2 = np.array(good_points_2)

    if len(good_points_1) < 4:
        # print("Not enough good points found for homography estimation.")
        return (None, None)
    # TODO: Test different thresholds, methods (USAC, RANSAC, ETC)
    H, mask = cv.findHomography(good_points_1, good_points_2, cv.USAC_MAGSAC, 0.25)
    # TODO: Do a local optimization on inlier points to improve the homography matrix
    if H is None:
        return (None, None)
    extracted_rotation = np.linalg.inv(cameraMatrix) @ H @ cameraMatrix
    U, _, Vt = np.linalg.svd(extracted_rotation)
    # orthonormalized_rotation = U @ Vt
    orthonormalized_rotation = extracted_rotation
    orthnormality = np.linalg.norm(
        orthonormalized_rotation @ orthonormalized_rotation.T - np.eye(3)
    )
    print(f"Orthonormality: {orthnormality:.4f}")

    inliers = np.sum(mask)
    # if inliers < 20:
    #     return (None, None)
    match_image = None
    if frame1 is not None:
        if frame2 is not None:
            match_image = np.hstack(
                (
                    cv.drawKeypoints(
                        frame1,
                        kp1,
                        None,
                        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                    ),
                    cv.drawKeypoints(
                        frame2,
                        kp2,
                        None,
                        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                    ),
                )
            )
        else:
            match_image = frame1.copy()
            for i in range(len(good_points_1)):
                if mask[i]:
                    pt1 = tuple(map(int, good_points_1[i]))
                    pt2 = tuple(map(int, good_points_2[i]))
                    cv.line(match_image, pt1, pt2, (0, 0, 255), 1)

    ret, rotations, translations, normals = cv.decomposeHomographyMat(H, cameraMatrix)
    # real_rot_mat = rotations[0]
    real_rot_mat = orthonormalized_rotation
    if ret == 0 or np.isnan(real_rot_mat).any() or np.isinf(real_rot_mat).any():
        return (None, match_image)

    # went from XYZ to ZXY
    reorder_mat = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    real_rot_mat = reorder_mat.T @ real_rot_mat @ reorder_mat
    # reverse roll
    if (
        np.linalg.det(real_rot_mat) < 0
    ):  # I don't know why non-positive determinants occur, but they do
        return (None, match_image)
    zxy = R.from_matrix(real_rot_mat).as_euler("ZXY", degrees=True)
    real_rot_mat = R.from_euler(
        "ZXY", [zxy[0], zxy[1], -zxy[2]], degrees=True
    ).as_matrix()

    return real_rot_mat, match_image


index_params = dict(algorithm=1, trees=5)
# 6 = FLANN_INDEX_LSH
# index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
get_angle_difference.flann = cv.FlannBasedMatcher(index_params, search_params)


def generate_rotation_histories_plot(
    rotation_histories, extra_text=None, extra_rot=None
):
    # # Create a blank figure above the main plot, with the same width
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
        vectors = [rotation @ np.array([0, 1, 0]) for rotation in rotations]
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
    ax_main.legend()
    ax_main.set_xticklabels([])
    ax_main.set_yticklabels([])
    ax_main.set_zticklabels([])

    # if len(rotation_histories[0]["data"]) == 48:
    #   # main_fig.show()
    #   plt.show(block=True)

    text_image = figure_to_cv_image(text_fig)
    main_image = figure_to_cv_image(main_fig)
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


# ZYX = YAW, PITCH, ROLL
# Yaw around Z, Pitch around Y, Roll around X


def Rz(alpha):
    return torch.tensor(
        [
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha), math.cos(alpha), 0],
            [0, 0, 1],
        ]
    )


def Ry(beta):
    return torch.tensor(
        [
            [math.cos(beta), 0, math.sin(beta)],
            [0, 1, 0],
            [-math.sin(beta), 0, math.cos(beta)],
        ]
    )


def Rx(gamma):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(gamma), -math.sin(gamma)],
            [0, math.sin(gamma), math.cos(gamma)],
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate angles from gyro data and video."
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("gyro_csv_path", type=str, help="Path to the gyro CSV file.")
    parser.add_argument(
        "output_angle_path", type=str, help="Path to the output angle CSV file."
    )
    parser.add_argument(
        "--output_video", type=str, help="Path to the output video file."
    )
    parser.add_argument(
        "--output_debug_video", type=str, help="Path to the debug output video file."
    )
    parser.add_argument(
        "--start_frame", type=int, default=0, help="Start frame for processing."
    )
    parser.add_argument(
        "--end_frame", type=int, default=-1, help="End frame for processing."
    )
    parser.add_argument(
        "--gopro", action="store_true", help="Use GoPro camera settings."
    )
    args = parser.parse_args()
    main(args)
