import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import config

# stereo_params = np.load(config.calib_dir + config.calib_sub_dir + 'stereo_params.npz')

# print(stereo_params['rotation_matrix'])
# print(Rotation.from_matrix(stereo_params['rotation_matrix']).as_euler('zyx', degrees=True))
# print(stereo_params['translation'])


unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

image_size = tuple(unrectify_params['image_size'])

left_rotation, right_rotation, left_projection, right_projection, _, _, _ = cv2.stereoRectify(unrectify_params['left_camera_matrix'], unrectify_params['left_distortion_coefficients'],
                                                                                              unrectify_params['right_camera_matrix'], unrectify_params['right_distortion_coefficients'],
                                                                                              image_size, np.eye(3), unrectify_params['translation'], flags=0)
# print(unrectify_params['left_camera_matrix'])
# print(unrectify_params['right_camera_matrix'])

# print(left_projection)
# print(right_projection)

# left_image = cv2.imread(config.calib_dir + config.calib_sub_dir + 'port_rect_3.tif')
# right_image = cv2.imread(config.calib_dir + config.calib_sub_dir + 'stbd_rect_3.tif')

# left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
# right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

# left_image[left_mask == 0] = 0
# right_image[right_mask == 0] = 0

# left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
# right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

# image_size = unrectify_params['image_size']

# left_mapx, left_mapy = cv2.initUndistortRectifyMap(unrectify_params['left_camera_matrix'], np.zeros(5), unrectify_params['left_rotation'], unrectify_params['left_projection'], image_size, cv2.CV_32F)
# right_mapx, right_mapy = cv2.initUndistortRectifyMap(unrectify_params['right_camera_matrix'], np.zeros(5), unrectify_params['right_rotation'], unrectify_params['right_projection'], image_size, cv2.CV_32F)

# left_image = cv2.remap(left_image, right_mapx, right_mapy, cv2.INTER_LINEAR)
# right_image = cv2.remap(right_image, left_mapx, left_mapy, cv2.INTER_LINEAR)

# plt.subplot(121)
# plt.imshow(left_image)
# plt.subplot(122)
# plt.imshow(right_image)
# plt.show()
