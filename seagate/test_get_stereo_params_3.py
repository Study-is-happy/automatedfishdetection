import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import config
import seagate_utils

match_points_dir = config.calib_dir + config.calib_sub_dir + 'test_match_points/'

all_left_points = []
all_right_points = []

test_left_points_list = []
test_right_points_list = []
test_match_points_files = []

for match_points_file in os.listdir(match_points_dir)[50:80]:

    match_points = np.load(match_points_dir + match_points_file)

    left_points = match_points['left_points']
    right_points = match_points['right_points']

    all_left_points.extend(left_points)
    all_right_points.extend(right_points)

    test_left_points_list.append(left_points)
    test_right_points_list.append(right_points)
    test_match_points_files.append(match_points_file)

all_left_points = np.array(all_left_points)
all_right_points = np.array(all_right_points)

print(len(all_left_points))

_, homography_mask = cv2.findHomography(all_left_points, all_right_points, cv2.RANSAC, 3.0)

homography_mask = (homography_mask.ravel() == 0)
all_left_points = all_left_points[homography_mask]
all_right_points = all_right_points[homography_mask]

print(len(all_left_points))

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']

norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((all_left_points.T, np.ones(len(all_left_points)))))[:-1].T
norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((all_right_points.T, np.ones(len(all_right_points)))))[:-1].T

_, essential_mask = cv2.findEssentialMat(norm_left_points, norm_right_points, np.eye(3), cv2.RANSAC, 0.999, 0.00005)

essential_mask = (essential_mask.ravel() == 1)
# essential_mask[12823] = 0
all_left_points = all_left_points[essential_mask]
all_right_points = all_right_points[essential_mask]
# 1261

print(len(all_left_points))

prior_stereo_params_path = config.calib_dir + config.calib_sub_dir + 'prior_stereo_params.npz'

if os.path.exists(prior_stereo_params_path):
    prior_stereo_params = np.load(prior_stereo_params_path)
    prior_rotation_matrix = prior_stereo_params['rotation_matrix']
    prior_translation = prior_stereo_params['translation']
else:
    prior_rotation_matrix = unrectify_params['opt_rotation_matrix']
    prior_translation = unrectify_params['opt_translation']

np.savez(config.calib_dir + config.calib_sub_dir + 'match_points',
         left_camera_matrix=left_camera_matrix,
         right_camera_matrix=right_camera_matrix,
         left_points=all_left_points,
         right_points=all_right_points,
         rotation_matrix=prior_rotation_matrix,
         translation=np.expand_dims(prior_translation, axis=1)
         )

test_mask = homography_mask.copy()
test_mask_indexes = np.where(test_mask)[0]
test_mask[test_mask_indexes] = essential_mask

test_seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-19-02/d20191102_17/'
test_left_dir = test_seagate_dir + 'port/port_rectified/'
test_right_dir = test_seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

current_index = 0
for test_match_points_file, test_left_points, test_right_points in zip(test_match_points_files, test_left_points_list, test_right_points_list):

    next_index = current_index + len(test_left_points)

    if 25489 >= current_index and 25489 < next_index:

        test_match_points_file = os.path.splitext(test_match_points_file)[0]
        left_image_id = test_match_points_file[:32]
        right_image_id = test_match_points_file[33:]

        test_left_points = test_left_points[25489 - current_index:25490 - current_index]
        test_right_points = test_right_points[25489 - current_index:25490 - current_index]

        # current_mask = test_mask[current_index:next_index]
        # test_left_points = test_left_points[current_mask]
        # test_right_points = test_right_points[current_mask]

        left_image = cv2.imread(test_left_dir + left_image_id + '.tif')
        right_image = cv2.imread(test_right_dir + right_image_id + '.tif')

        left_image[left_mask == 0] = 0
        right_image[right_mask == 0] = 0

        left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
        right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

        seagate_utils.plot_match_points(left_image, right_image, test_left_points, test_right_points)

    current_index = next_index
