import numpy as np
import cv2
import os
import glob
import random
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import utils
import seagate_utils
import config

match_points_dir = config.calib_dir + config.calib_sub_dir + 'filtered_match_points/'

match_points_list = sorted(os.listdir(match_points_dir))

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']
image_size = tuple(unrectify_params['image_size'])

seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-19-02/d20191009_1/'
left_dir = seagate_dir + 'port/port_rectified/'
right_dir = seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

stereo_params_dir = config.calib_dir + config.calib_sub_dir + 'stereo_params/'

for match_points_file in match_points_list:

    stereo_params = np.load(stereo_params_dir + match_points_file)
    # stereo_params = np.load(config.calib_dir + config.calib_sub_dir + 'global_stereo_params.npz')

    match_points = np.load(match_points_dir + match_points_file)

    left_points = match_points['left_points']
    right_points = match_points['right_points']

    left_image_id, right_image_id = os.path.splitext(match_points_file)[0].split('=')
    print(left_image_id)

    # if left_image_path != left_dir + '20181012.181524.00070_rect_color.tif':
    #     continue

    left_image = cv2.imread(left_dir + left_image_id + '.tif')
    right_image = cv2.imread(right_dir + right_image_id + '.tif')

    left_image[left_mask == 0] = 0
    right_image[right_mask == 0] = 0

    left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
    right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

    # seagate_utils.plot_match_points(left_image, right_image, left_points, right_points, True)

    left_image = cv2.remap(left_image, stereo_params['left_mapx'], stereo_params['left_mapy'], cv2.INTER_LINEAR)
    right_image = cv2.remap(right_image, stereo_params['right_mapx'], stereo_params['right_mapy'], cv2.INTER_LINEAR)

    left_points = np.squeeze(cv2.undistortPoints(left_points,
                                                 left_camera_matrix,
                                                 np.zeros(5),
                                                 R=stereo_params['left_rotation'],
                                                 P=stereo_params['left_projection']))

    right_points = np.squeeze(cv2.undistortPoints(right_points,
                                                  right_camera_matrix,
                                                  np.zeros(5),
                                                  R=stereo_params['right_rotation'],
                                                  P=stereo_params['right_projection']))

    points_diff = left_points - right_points

    points_diff = points_diff[:, 0][np.abs(points_diff[:, 1]) < 0.5]

    disparity_unit = 16
    min_disparity = utils.get_rint(np.min(points_diff)) - disparity_unit
    max_disparity = utils.get_rint(np.max(points_diff)) + disparity_unit

    print(min_disparity, max_disparity)

    num_disparities = max_disparity - min_disparity
    num_disparities += disparity_unit - num_disparities % disparity_unit

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
