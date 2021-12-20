import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

import config
import seagate_utils

match_points_dir = config.calib_dir + config.calib_sub_dir + 'all_match_points/'

all_left_points = []
all_right_points = []

for match_points_file in sorted(os.listdir(match_points_dir)):

    match_points = np.load(match_points_dir + match_points_file)

    left_points = match_points['left_points']
    right_points = match_points['right_points']

    all_left_points.extend(left_points)
    all_right_points.extend(right_points)

all_left_points = np.array(all_left_points)
all_right_points = np.array(all_right_points)

print(len(all_left_points))

_, homography_mask = cv2.findHomography(all_left_points, all_right_points, cv2.RANSAC, 1.0, None, 100000, 0.9999)
_, homography_mask = cv2.findHomography(all_left_points, all_right_points, cv2.RANSAC, 1.0, None, 100000, 0.9999)

homography_mask = (homography_mask.ravel() == 1)
all_left_points = all_left_points[homography_mask]
all_right_points = all_right_points[homography_mask]

print(len(all_left_points))

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']

norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((all_left_points.T, np.ones(len(all_left_points)))))[:-1].T
norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((all_right_points.T, np.ones(len(all_right_points)))))[:-1].T

homography_matrix, _ = cv2.findHomography(norm_left_points, norm_right_points)
_, rotation_matrix_list, translation_list, _ = cv2.decomposeHomographyMat(homography_matrix, np.eye(3))

best_error = np.inf
for index, translation in enumerate(translation_list):

    translation = translation.flatten()
    translation = translation * cv2.norm(unrectify_params['opt_translation'][:2]) / cv2.norm(translation[:2])
    error = cv2.norm(translation - unrectify_params['opt_translation'])
    if error < best_error:
        best_error = error
        best_index = index
        best_translation = translation

rotation_matrix = rotation_matrix_list[best_index]
translation = best_translation
print(translation)

image_size = tuple(unrectify_params['image_size'])

left_rotation, right_rotation, left_projection, right_projection, Q, left_roi, right_roi = cv2.stereoRectify(left_camera_matrix, np.zeros(5),
                                                                                                             right_camera_matrix, np.zeros(5),
                                                                                                             image_size, rotation_matrix, translation)

left_mapx, left_mapy = cv2.initUndistortRectifyMap(left_camera_matrix, np.zeros(5), left_rotation, left_projection, image_size, cv2.CV_32F)
right_mapx, right_mapy = cv2.initUndistortRectifyMap(right_camera_matrix, np.zeros(5), right_rotation, right_projection, image_size, cv2.CV_32F)

np.savez(config.calib_dir + config.calib_sub_dir + 'stereo_params',
         rotation_matrix=rotation_matrix,
         translation=translation,
         left_rotation=left_rotation,
         right_rotation=right_rotation,
         left_projection=left_projection,
         right_projection=right_projection,
         Q=Q,
         left_mapx=left_mapx,
         left_mapy=left_mapy,
         right_mapx=right_mapx,
         right_mapy=right_mapy,
         left_roi=left_roi,
         right_roi=right_roi)
