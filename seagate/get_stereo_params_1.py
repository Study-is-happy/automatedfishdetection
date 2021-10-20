import numpy as np
import cv2
import os

import config

match_points_dir = config.calib_dir + config.calib_sub_dir + 'match_points/'

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

print(len(all_left_points))

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']

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
