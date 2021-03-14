import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import config

calib_file = cv2.FileStorage(config.calib_dir + 'calib.xml', cv2.FILE_STORAGE_WRITE)

image_width = 2448
image_height = 2050

calib_file.write('image_width', image_width)
calib_file.write('image_height', image_height)

left_camera_matrix = np.array([[3180.67562, 0, 1236.33861],
                               [0, 3186.40381, 1019.86792],
                               [0, 0, 1]])

calib_file.write('left_camera_matrix', left_camera_matrix)

left_distortion_coefficients = np.array([0.17105, 0.55328, 0.00347, -0.00242])

calib_file.write('left_distortion_coefficients', left_distortion_coefficients)

right_camera_matrix = np.array([[3179.79977, 0, 1236.50966],
                                [0, 3184.83088, 1013.60488],
                                [0, 0, 1]])

calib_file.write('right_camera_matrix', right_camera_matrix)

right_distortion_coefficients = np.array([0.17161, 0.59247, 0.00262, -0.00213])

calib_file.write('right_distortion_coefficients', right_distortion_coefficients)

translation = np.array([-181.81267, 4.49545, 1.15069])
rotation_matrix = Rotation.from_rotvec([-0.00466, 0.00723, 0.02574]).as_matrix()

calib_file.write('translation', translation)
calib_file.write('rotation_matrix', rotation_matrix)

calib_file.release()
