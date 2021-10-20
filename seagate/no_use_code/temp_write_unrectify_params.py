import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import config

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

image_size = tuple(unrectify_params['image_size'])

_, left_roi = cv2.getOptimalNewCameraMatrix(unrectify_params['left_camera_matrix'], unrectify_params['left_distortion_coefficients'], image_size, 1, image_size)
_, right_roi = cv2.getOptimalNewCameraMatrix(unrectify_params['right_camera_matrix'], unrectify_params['right_distortion_coefficients'], image_size, 1, image_size)

np.savez(config.calib_dir + 'unrectify_params',
         image_size=image_size,
         left_camera_matrix=unrectify_params['left_camera_matrix'],
         right_camera_matrix=unrectify_params['right_camera_matrix'],
         left_distortion_coefficients=unrectify_params['left_distortion_coefficients'],
         right_distortion_coefficients=unrectify_params['right_distortion_coefficients'],
         rotation_matrix=unrectify_params['rotation_matrix'],
         translation=unrectify_params['translation'],
         left_rotation=unrectify_params['left_rotation'],
         right_rotation=unrectify_params['right_rotation'],
         left_projection=unrectify_params['left_projection'],
         right_projection=unrectify_params['right_projection'],
         opt_rotation_matrix=unrectify_params['opt_rotation_matrix'],
         opt_translation=unrectify_params['opt_translation'],
         inv_left_mapx=unrectify_params['inv_left_mapx'],
         inv_left_mapy=unrectify_params['inv_left_mapy'],
         inv_right_mapx=unrectify_params['inv_right_mapx'],
         inv_right_mapy=unrectify_params['inv_right_mapy'],
         left_roi=left_roi,
         right_roi=right_roi)
