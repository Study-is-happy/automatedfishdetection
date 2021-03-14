import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import util

width = 2448
height = 2050

left_camera_matrix = np.array([[3200.30144, 0, 1238.67489],
                               [0, 3194.1071, 991.12046],
                               [0, 0, 1]])

left_distortion_coefficients = np.array([0.17248, 0.57233, 0.00131, -0.00236])

right_camera_matrix = np.array([[3210.14443, 0, 1238.01240],
                                [0, 3204.52724, 1009.88894],
                                [0, 0, 1]])

right_distortion_coefficients = np.array([0.15904, 0.71345, 0.00190, -0.00263])

translation = np.array([-181.92837, 3.53263, 6.22448])
rotation_matrix = Rotation.from_rotvec([-0.00313, 0.01175, 0.02402]).as_matrix()

left_rotation, right_rotation, left_projection, right_projection, Q, left_roi, right_roi = cv2.stereoRectify(left_camera_matrix, left_distortion_coefficients,
                                                                                                             right_camera_matrix, right_distortion_coefficients,
                                                                                                             (width, height), rotation_matrix, translation, flags=0)

with open('/home/auv/venv3/left_projection.npy', 'wb') as left_projection_file:
    np.save(left_projection_file, left_projection)

with open('/home/auv/venv3/left_rotation.npy', 'wb') as left_rotation_file:
    np.save(left_rotation_file, left_rotation)

with open('/home/auv/venv3/left_camera_matrix.npy', 'wb') as left_camera_matrix_file:
    np.save(left_camera_matrix_file, left_camera_matrix)

inv_mapx, inv_mapy = cv2.initUndistortRectifyMap(left_projection[:, :-1], np.zeros(5), np.linalg.inv(left_rotation), left_camera_matrix, (width, height), cv2.CV_32F)
# inv_mapx, inv_mapy = cv2.initUndistortRectifyMap(right_projection[:, :-1], np.zeros(5), np.linalg.inv(right_rotation), right_camera_matrix, (width, height), cv2.CV_32F)

image = cv2.imread('/home/auv/venv3/port_2_rect.jpg')

image = cv2.remap(image, inv_mapx, inv_mapy, cv2.INTER_LINEAR)

mask_image = cv2.imread('/home/auv/venv3/port_mask.bmp', -1)
mask_image[mask_image == 255] = 0

# with open('/home/auv/venv3/instances.json') as instances_file:

#     instances_dict = json.load(instances_file)
#     instance = instances_dict['20161027.175018.00256_rect_color']

#     for annotation in instance['annotations']:
#         bbox = annotation['bbox']
#         util.rel_to_abs(bbox, instance['width'], instance['height'])

#         bbox = np.rint(bbox).astype(int)

#         cv2.rectangle(mask_image, tuple(bbox[0:2]), tuple(bbox[2:4]), 255, -1)


mask_image = cv2.remap(mask_image, inv_mapx, inv_mapy, cv2.INTER_LINEAR)

mask_image[mask_image < 127] = 0
image[mask_image == 0] = 0

roi_x, roi_y, roi_w, roi_h = left_roi
mask_roi_image = mask_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
mask_roi_image[mask_roi_image == 127] = 255
mask_image[mask_image != 255] = 0

plt.imshow(mask_image)
plt.show()

# cv2.imwrite('/home/auv/venv3/port_2.png', image)
# cv2.imwrite('/home/auv/venv3/port_mask.png', mask_image)
plt.imshow(image)
plt.show()
