import numpy as np
import cv2
import os
import glob
import random
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import seagate_utils
import config

random.seed(0)

match_points_dir = config.calib_dir + config.calib_sub_dir + 'all_match_points/'

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']
image_size = tuple(unrectify_params['image_size'])

norm_opt_translation = unrectify_params['opt_translation'] / cv2.norm(unrectify_params['opt_translation'])
scale = cv2.norm(unrectify_params['opt_translation'][:2])

all_left_points = []
all_right_points = []

for match_points_file in os.listdir(match_points_dir):

    match_points = np.load(match_points_dir + match_points_file)

    left_points = match_points['left_points']
    right_points = match_points['right_points']

    all_left_points.extend(left_points)
    all_right_points.extend(right_points)

all_left_points = np.array(all_left_points)
all_right_points = np.array(all_right_points)

_, homography_mask = cv2.findHomography(all_left_points, all_right_points, cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=100000, confidence=0.9999)
fundamental_matrix, fundamental_mask = cv2.findFundamentalMat(all_left_points, all_right_points, cv2.FM_RANSAC, 0.2)

homography_inlier_count = np.count_nonzero(homography_mask)
fundamental_inlier_count = np.count_nonzero(fundamental_mask)

print(homography_inlier_count, fundamental_inlier_count)

if homography_inlier_count > fundamental_inlier_count:

    homography_mask = (homography_mask.ravel() == 1)
    inlier_all_left_points = all_left_points[homography_mask]
    inlier_all_right_points = all_right_points[homography_mask]

    norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((inlier_all_left_points.T, np.ones(len(inlier_all_left_points)))))[:-1].T
    norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((inlier_all_right_points.T, np.ones(len(inlier_all_right_points)))))[:-1].T

    homography_matrix, _ = cv2.findHomography(norm_left_points, norm_right_points)
    _, rotation_matrix_list, translation_list, _ = cv2.decomposeHomographyMat(homography_matrix, np.eye(3))

    best_error = np.inf
    for transform_index, translation in enumerate(translation_list):

        error = cv2.norm(translation.flatten() / cv2.norm(translation) - norm_opt_translation)

        if error < best_error:
            best_error = error
            best_transform_index = transform_index
            best_translation = translation

    rotation_matrix = rotation_matrix_list[best_transform_index]
    translation = best_translation

else:

    norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((all_left_points.T, np.ones(len(all_left_points)))))[:-1].T
    norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((all_right_points.T, np.ones(len(all_right_points)))))[:-1].T

    len_match_points = len(all_left_points)

    while True:

        random_indexes = random.sample(range(len_match_points), len_match_points)

        all_left_points = all_left_points[random_indexes]
        all_right_points = all_right_points[random_indexes]
        norm_left_points = norm_left_points[random_indexes]
        norm_right_points = norm_right_points[random_indexes]

        essential_matrix, essential_mask = cv2.findEssentialMat(norm_left_points, norm_right_points, np.eye(3), cv2.RANSAC, 0.999, 0.0005)

        essential_mask = (essential_mask.ravel() == 1)
        inlier_all_left_points = all_left_points[essential_mask]
        inlier_all_right_points = all_right_points[essential_mask]
        inlier_norm_left_points = norm_left_points[essential_mask]
        inlier_norm_right_points = norm_right_points[essential_mask]

        _, rotation_matrix, translation, _ = cv2.recoverPose(essential_matrix, inlier_norm_left_points, inlier_norm_right_points)

        # left_projection_matrix = left_camera_matrix.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
        # right_projection_matrix = right_camera_matrix.dot(np.hstack((rotation_matrix, translation)))

        # object_points = cv2.triangulatePoints(
        #     left_projection_matrix, right_projection_matrix, inlier_all_left_points.T, inlier_all_right_points.T)
        # object_points = (object_points / object_points[-1])[:-1].T

        # object_points_depth = object_points[:, -1]

        # histogram, bin_edges = np.histogram(object_points_depth, bins=np.arange(np.min(object_points_depth), np.max(object_points_depth) + 2, 2))

        # for left_index, in enumerate(histogram)

        # print(np.count_nonzero(reprojection_mask), np.count_nonzero(essential_mask))
        # if np.count_nonzero(reprojection_mask) == np.count_nonzero(essential_mask):

        # print(left_reproject_errors[~inlier_mask], right_reproject_errors[~inlier_mask])

        # test_mask = object_points[:, -1] > 10

        # inlier_norm_left_points = inlier_norm_left_points[test_mask]
        # inlier_norm_right_points = inlier_norm_right_points[test_mask]

        # inlier_all_left_points = inlier_all_left_points[test_mask]
        # inlier_all_right_points = inlier_all_right_points[test_mask]

        # essential_matrix, _ = cv2.findEssentialMat(inlier_norm_left_points, inlier_norm_right_points, np.eye(3), cv2.RANSAC, 0.999, 0.0005)
        # _, rotation_matrix, translation, _ = cv2.recoverPose(essential_matrix, inlier_norm_left_points, inlier_norm_right_points)

        break

translation = translation.flatten()
translation = translation / cv2.norm(translation[:2]) * scale

print(Rotation.from_matrix(rotation_matrix).as_euler('zyx', degrees=True))
print(translation)

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

np.savez(config.calib_dir + config.calib_sub_dir + 'match_points',
         left_camera_matrix=left_camera_matrix,
         right_camera_matrix=right_camera_matrix,
         left_points=inlier_all_left_points,
         right_points=inlier_all_right_points,
         rotation_matrix=rotation_matrix,
         translation=np.expand_dims(translation, axis=1)
         )
