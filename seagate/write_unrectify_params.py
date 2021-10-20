import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import config

image_width = 2448
image_height = 2050

image_size = (image_width, image_height)

left_camera_matrix = np.array([[3175.48536, 0, 1247.54014],
                               [0, 3180.45659, 1017.97918],
                               [0, 0, 1]])

left_distortion_coefficients = np.array([0.16788, 0.57792, 0.00263, -0.00257])

right_camera_matrix = np.array([[3178.36059, 0, 1227.24350],
                                [0, 3183.22266, 1007.84377],
                                [0, 0, 1]])

right_distortion_coefficients = np.array([0.15558, 0.68340, 0.00141, -0.00358])

rotation_matrix = Rotation.from_rotvec([-0.00816, 0.01059, 0.02541]).as_matrix()
translation = np.array([-181.89460, -6.82327, 3.52862])

left_rotation, right_rotation, left_projection, right_projection, _, _, _ = cv2.stereoRectify(left_camera_matrix, left_distortion_coefficients,
                                                                                              right_camera_matrix, right_distortion_coefficients,
                                                                                              image_size, rotation_matrix, translation, flags=0)


inv_left_mapx, inv_left_mapy = cv2.initUndistortRectifyMap(left_projection[:, :-1], np.zeros(5), np.linalg.inv(left_rotation), left_camera_matrix, image_size, cv2.CV_32F)
inv_right_mapx, inv_right_mapy = cv2.initUndistortRectifyMap(right_projection[:, :-1], np.zeros(5), np.linalg.inv(right_rotation), right_camera_matrix, image_size, cv2.CV_32F)

# opt_left_camera_matrix = np.array([[3204.24223, 0, 1238.56828],
#                                [0, 3198.06956, 999.80362],
#                                [0, 0, 1]])

# opt_left_distortion_coefficients = np.array([0.17550, 0.56077, 0.00259, -0.00204])

# opt_right_camera_matrix = np.array([[3206.35808, 0, 1235.67787],
#                                 [0, 3199.78031, 1003.55245],
#                                 [0, 0, 1]])

# opt_right_distortion_coefficients = np.array([0.15850, 0.70459, 0.00057, -0.00273])

opt_rotation_matrix = Rotation.from_rotvec([-0.00816, 0.01059, 0.02541]).as_matrix()
opt_translation = np.array([-181.89460, -6.82327, 3.52862])

_, left_roi = cv2.getOptimalNewCameraMatrix(left_camera_matrix, left_distortion_coefficients, image_size, 1, image_size)
_, right_roi = cv2.getOptimalNewCameraMatrix(right_camera_matrix, right_distortion_coefficients, image_size, 1, image_size)

np.savez(config.calib_dir + 'unrectify_params',
         image_size=image_size,
         left_camera_matrix=left_camera_matrix,
         right_camera_matrix=right_camera_matrix,
         left_distortion_coefficients=left_distortion_coefficients,
         right_distortion_coefficients=right_distortion_coefficients,
         rotation_matrix=rotation_matrix,
         translation=translation,
         left_rotation=left_rotation,
         right_rotation=right_rotation,
         left_projection=left_projection,
         right_projection=right_projection,
         opt_rotation_matrix=opt_rotation_matrix,
         opt_translation=opt_translation,
         inv_left_mapx=inv_left_mapx,
         inv_left_mapy=inv_left_mapy,
         inv_right_mapx=inv_right_mapx,
         inv_right_mapy=inv_right_mapy,
         left_roi=left_roi,
         right_roi=right_roi)
