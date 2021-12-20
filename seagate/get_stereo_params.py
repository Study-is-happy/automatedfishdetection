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

match_points_dir = config.calib_dir + config.calib_sub_dir + 'sift_match_points/'

match_points_file_list = sorted(os.listdir(all_match_points_dir))

half_near_num = 5

for match_index, match_points_file in enumerate(match_points_file_list):

    left_near_index = max(match_index - half_near_num, 0)
    right_near_index = min(match_index + half_near_num + 1, len(match_points_file_list))

    near_match_points_list = match_points_file_list[left_near_index:right_near_index]

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
