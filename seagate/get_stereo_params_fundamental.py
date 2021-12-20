import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import config
import seagate_utils

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']
scale = cv2.norm(unrectify_params['opt_translation'][:2])

all_match_points_dir = config.calib_dir + config.calib_sub_dir + 'sift_match_points/'

all_match_points_file_list = sorted(os.listdir(all_match_points_dir))

all_left_points = []
all_right_points = []
all_points_mask = []

for index, match_points_file in enumerate(all_match_points_file_list):

    match_points = np.load(all_match_points_dir + match_points_file)

    left_points = match_points['left_points']
    right_points = match_points['right_points']

    all_left_points.extend(left_points)
    all_right_points.extend(right_points)
    all_points_mask.extend(np.ones(len(left_points), dtype=int) * index)

all_left_points = np.array(all_left_points)
all_right_points = np.array(all_right_points)
all_points_mask = np.array(all_points_mask)

print(len(all_left_points))

# _, homography_mask = cv2.findHomography(all_left_points, all_right_points, cv2.RANSAC, 2.0, None, 100000, 0.9999)

# homography_mask = (homography_mask.ravel() == 0)
# all_left_points = all_left_points[homography_mask]
# all_right_points = all_right_points[homography_mask]

# print(len(all_left_points))

norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((all_left_points.T, np.ones(len(all_left_points)))))[:-1].T
norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((all_right_points.T, np.ones(len(all_right_points)))))[:-1].T

essential_matrix, essential_mask = cv2.findEssentialMat(norm_left_points, norm_right_points, np.eye(3), cv2.RANSAC, prob=0.9999, threshold=0.0005, maxIters=100000)
# essential_matrix, essential_mask = cv2.findEssentialMat(all_left_points, all_right_points, left_camera_matrix, right_camera_matrix, np.zeros(5), np.zeros(5), cv2.RANSAC, prob=0.9999, threshold=1.0)
essential_mask = (essential_mask.ravel() == 1)
norm_left_points = norm_left_points[essential_mask]
norm_right_points = norm_right_points[essential_mask]
all_left_points = all_left_points[essential_mask]
all_right_points = all_right_points[essential_mask]
all_points_mask = all_points_mask[essential_mask]
print(len(norm_left_points))

_, rotation_matrix, translation, recover_pose_mask = cv2.recoverPose(essential_matrix, norm_left_points, norm_right_points)
recover_pose_mask = (recover_pose_mask.ravel() == 255)
all_left_points = all_left_points[recover_pose_mask]
all_right_points = all_right_points[recover_pose_mask]
all_points_mask = all_points_mask[recover_pose_mask]

translation = translation.flatten()
translation = translation / cv2.norm(translation[:2]) * scale

print(translation)

image_size = tuple(unrectify_params['image_size'])

left_rotation, right_rotation, left_projection, right_projection, Q, left_roi, right_roi = cv2.stereoRectify(left_camera_matrix, np.zeros(5),
                                                                                                             right_camera_matrix, np.zeros(5),
                                                                                                             image_size, rotation_matrix, translation)

left_mapx, left_mapy = cv2.initUndistortRectifyMap(left_camera_matrix, np.zeros(5), left_rotation, left_projection, image_size, cv2.CV_32F)
right_mapx, right_mapy = cv2.initUndistortRectifyMap(right_camera_matrix, np.zeros(5), right_rotation, right_projection, image_size, cv2.CV_32F)

np.savez(config.calib_dir + config.calib_sub_dir + 'global_stereo_params',
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

filtered_match_points_dir = config.calib_dir + config.calib_sub_dir + 'filtered_match_points/'


for index, match_points_file in enumerate(all_match_points_file_list):
    np.savez(filtered_match_points_dir + match_points_file,
             left_points=all_left_points[all_points_mask == index],
             right_points=all_right_points[all_points_mask == index])
