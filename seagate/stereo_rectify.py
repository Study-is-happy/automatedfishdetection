import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

left_image = cv2.imread('/home/auv/venv3/stbd_2.png')
right_image = cv2.imread('/home/auv/venv3/port_2.png')
plt.imshow(left_image)
plt.show()

height, width, _ = left_image.shape

left_camera_matrix = np.array([[3209.85936, 0, 1254.70681],
                               [0, 3204.15839, 1018.59789],
                               [0, 0, 1]])

left_distortion_coefficients = np.array([0.17646, 0.58992, 0.00592, 0.00078])

right_camera_matrix = np.array([[3212.80907, 0, 1245.25003],
                                [0, 3205.90827, 1019.85816],
                                [0, 0, 1]])

right_distortion_coefficients = np.array([0.16445, 0.68139, 0.00354, -0.00109])

translation = np.array([-181.24763, 4.11051, 1.81262])
rotation_matrix = Rotation.from_rotvec([-0.00827, 0.01406, 0.02427]).as_matrix()

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion_coefficients,
                                                                  right_camera_matrix, right_distortion_coefficients,
                                                                  (width, height), rotation_matrix, translation)

left_mapx, left_mapy = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion_coefficients, R1, P1, (width, height), cv2.CV_32F)
right_mapx, right_mapy = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion_coefficients, R2, P2, (width, height), cv2.CV_32F)

left_image = cv2.remap(left_image, left_mapx, left_mapy, cv2.INTER_LINEAR)
right_image = cv2.remap(right_image, right_mapx, right_mapy, cv2.INTER_LINEAR)

cv2.imwrite('/home/auv/venv3/port_2_my_rect.png', left_image)
cv2.imwrite('/home/auv/venv3/stbd_2_my_rect.png', right_image)
plt.imshow(left_image)
plt.show()
