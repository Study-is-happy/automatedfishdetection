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

match_points_list = sorted(os.listdir(match_points_dir))

half_near_num = 0

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']
image_size = tuple(unrectify_params['image_size'])

norm_opt_translation = unrectify_params['opt_translation'] / cv2.norm(unrectify_params['opt_translation'])
scale = cv2.norm(unrectify_params['opt_translation'][:2])

seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/SH-18-12/d20181015_3/'
left_dir = seagate_dir + 'port/port_rectified/'
right_dir = seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

for match_index, match_points_file in enumerate(match_points_list):

    if match_points_file != '20181015.155917.00136_rect_color=20181015.155917.00135_rect_color.npz':
        continue

    left_near_index = max(match_index - half_near_num, 0)
    right_near_index = min(match_index + half_near_num + 1, len(match_points_list))

    near_match_points_list = match_points_list[left_near_index:right_near_index]

    near_left_points = []
    near_right_points = []

    for near_match_points_file in near_match_points_list:

        match_points = np.load(match_points_dir + near_match_points_file)
        print(match_points_file)

        left_points = match_points['left_points']
        right_points = match_points['right_points']

        near_left_points.extend(left_points)
        near_right_points.extend(right_points)

    near_left_points = np.array(near_left_points)
    near_right_points = np.array(near_right_points)

    _, homography_mask = cv2.findHomography(near_left_points, near_right_points, cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=100000, confidence=0.9999)
    fundamental_matrix, fundamental_mask = cv2.findFundamentalMat(near_left_points, near_right_points, cv2.FM_RANSAC, 0.4)

    homography_inlier_count = np.count_nonzero(homography_mask)
    fundamental_inlier_count = np.count_nonzero(fundamental_mask)

    print(homography_inlier_count, fundamental_inlier_count)

    if homography_inlier_count > fundamental_inlier_count:

        homography_mask = (homography_mask.ravel() == 1)
        inlier_near_left_points = near_left_points[homography_mask]
        inlier_near_right_points = near_right_points[homography_mask]

        norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((inlier_near_left_points.T, np.ones(len(inlier_near_left_points)))))[:-1].T
        norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((inlier_near_right_points.T, np.ones(len(inlier_near_right_points)))))[:-1].T

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

        norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((near_left_points.T, np.ones(len(near_left_points)))))[:-1].T
        norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((near_right_points.T, np.ones(len(near_right_points)))))[:-1].T

        essential_matrix, essential_mask = cv2.findEssentialMat(norm_left_points, norm_right_points, np.eye(3), cv2.RANSAC, 0.999, 0.0005)

        essential_mask = (essential_mask.ravel() == 1)
        inlier_near_left_points = near_left_points[essential_mask]
        inlier_near_right_points = near_right_points[essential_mask]
        inlier_norm_left_points = norm_left_points[essential_mask]
        inlier_norm_right_points = norm_right_points[essential_mask]

        _, rotation_matrix, translation, _ = cv2.recoverPose(essential_matrix, inlier_norm_left_points, inlier_norm_right_points)

    translation = translation.flatten()
    translation = translation / cv2.norm(translation[:2]) * scale

    print(Rotation.from_matrix(rotation_matrix).as_euler('zyx', degrees=True))
    print(translation)

    left_rotation, right_rotation, left_projection, right_projection, Q, left_roi, right_roi = cv2.stereoRectify(left_camera_matrix, np.zeros(5),
                                                                                                                 right_camera_matrix, np.zeros(5),
                                                                                                                 image_size, rotation_matrix, translation)

    left_mapx, left_mapy = cv2.initUndistortRectifyMap(left_camera_matrix, np.zeros(5), left_rotation, left_projection, image_size, cv2.CV_32F)
    right_mapx, right_mapy = cv2.initUndistortRectifyMap(right_camera_matrix, np.zeros(5), right_rotation, right_projection, image_size, cv2.CV_32F)

    np.savez(config.calib_dir + config.calib_sub_dir + 'stereo_params/' + match_points_file,
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

    left_image_id, right_image_id = os.path.splitext(match_points_file)[0].split('=')

    left_image = cv2.imread(left_dir + left_image_id + '.tif')
    right_image = cv2.imread(right_dir + right_image_id + '.tif')

    left_image[left_mask == 0] = 0
    right_image[right_mask == 0] = 0

    left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
    right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

    seagate_utils.plot_match_points(left_image, right_image, inlier_near_left_points, inlier_near_right_points, True)

    left_image = cv2.remap(left_image, left_mapx, left_mapy, cv2.INTER_LINEAR)
    right_image = cv2.remap(right_image, right_mapx, right_mapy, cv2.INTER_LINEAR)

    inlier_near_left_points = np.squeeze(cv2.undistortPoints(inlier_near_left_points,
                                                             left_camera_matrix,
                                                             np.zeros(5),
                                                             R=left_rotation,
                                                             P=left_projection))

    inlier_near_right_points = np.squeeze(cv2.undistortPoints(inlier_near_right_points,
                                                              right_camera_matrix,
                                                              np.zeros(5),
                                                              R=right_rotation,
                                                              P=right_projection))

    match_disparity = np.sqrt(np.sum((inlier_near_left_points - inlier_near_right_points)**2, axis=1))

    disparity_unit = 16
    min_disparity = int(np.min(match_disparity)) - disparity_unit
    num_disparities = int(np.max(match_disparity)) + disparity_unit * 2 - min_disparity
    num_disparities += 16 - num_disparities % disparity_unit

    print(np.min(match_disparity))
    print(np.max(match_disparity))

    stereo_disparity = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                             numDisparities=num_disparities,
                                             blockSize=5,
                                             P1=8 * 3 * 5**2,
                                             P2=32 * 3 * 5**2,
                                             # disp12MaxDiff=0,
                                             # preFilterCap=32,
                                             # uniquenessRatio=15,
                                             mode=cv2.STEREO_SGBM_MODE_HH
                                             )

    disparity_map = np.float32(stereo_disparity.compute(left_image, right_image)) / disparity_unit

    disparity_image = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparity_image = cv2.cvtColor(disparity_image, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(config.calib_dir + config.calib_sub_dir + 'disparity_images/' + left_image_id + '.png', disparity_image)

    plt.subplot(121)
    plt.imshow(left_image)
    plt.subplot(122)
    plt.imshow(disparity_image)
    plt.show()
