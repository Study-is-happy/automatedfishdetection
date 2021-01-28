import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

width = 2448
height = 2050

image = cv2.imread('/home/auv/venv3/calib_stbd.jpg')
plt.imshow(image)
plt.show()

# left_camera_matrix = np.array([[3209.85936, 0, 1254.70681],
#                                [0, 3204.15839, 1018.59789],
#                                [0, 0, 1]])

# left_distortion_coefficients = np.array([0.17646, 0.58992, 0.00592, 0.00078])

# right_camera_matrix = np.array([[3212.80907, 0, 1245.25003],
#                                 [0, 3205.90827, 1019.85816],
#                                 [0, 0, 1]])

# right_distortion_coefficients = np.array([0.16445, 0.68139, 0.00354, -0.00109])

# translation = np.array([-181.24763, 4.11051, 1.81262])
# rotation_matrix = Rotation.from_rotvec([-0.00827, 0.01406, 0.02427]).as_matrix()

left_camera_matrix = np.array([[3209.85936, 0, 1249.95221995823],
                               [0, 3204.15839, 992.647861240359],
                               [0, 0, 1]])

left_distortion_coefficients = np.array([0.184638858817734, 0.431666696846974, 0.00100070545915401, -0.000453504496674385, 0.663453177514042])

right_camera_matrix = np.array([[3188.48892169631, 0, 1243.32214546065],
                                [0, 3182.85643350219, 993.531258472981],
                                [0, 0, 1]])

right_distortion_coefficients = np.array([0.186305074613455, 0.276472991951693, -0.000731128886409580, -0.00244740261782526, 1.60541578358303])

translation = np.array([-89.2559349198604, 2.16300194058934, -2.60748191579425])
rotation_matrix = np.array([[0.999625891517368, 0.0237665461376619, -0.0135361845729454],
                            [-0.0238764424246033, 0.999682781053290, -0.00801578210191261],
                            [0.0133413831835782, 0.00833597926143812, 0.999876251815444]])


R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion_coefficients,
                                                                  right_camera_matrix, right_distortion_coefficients,
                                                                  (width, height), rotation_matrix, translation, flags=0)

mapx, mapy = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion_coefficients, R2, P2, (width, height), cv2.CV_32F)
image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

cv2.imwrite('/home/auv/venv3/calib_stbd_vik_rect.jpg', image)
plt.imshow(image)
plt.show()
