import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

width = 2448
height = 2050

mask_image = np.full((height, width), 127, np.uint8)

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

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion_coefficients,
                                                                  right_camera_matrix, right_distortion_coefficients,
                                                                  (width, height), rotation_matrix, translation, flags=0)


# mapx, mapy = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion_coefficients, R1, P1, (width, height), cv2.CV_32F)
mapx, mapy = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion_coefficients, R1, P1, (width, height), cv2.CV_16SC2)

mask_image = cv2.remap(mask_image, mapx, mapy, cv2.INTER_LINEAR)

image = cv2.imread('/home/auv/venv3/stbd.jpg', -1)

orig_image = np.zeros((height, width, 3), np.uint8)

for i in range(width):
    for j in range(height):
        map_xy = mapx[j, i]
        map_i = map_xy[0]
        map_j = map_xy[1]
        if map_i >= 0 and map_j >= 0 and map_i < width and map_j < height:
            orig_image[map_j, map_i, :] = image[j, i, :]

# matlab_mask_image = cv2.imread('/home/auv/venv3/port_mask.bmp', -1)
# matlab_mask_image[matlab_mask_image == 255] = 0

cv2.imwrite('/home/auv/venv3/orig_stbd.jpg', orig_image)
plt.imshow(orig_image)
plt.show()
