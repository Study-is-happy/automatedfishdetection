import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import config

transformation_matrix = np.load(config.calib_dir + config.calib_sub_dir + 'transformation_matrix.npy')

rotation_matrix = transformation_matrix[:3, :3]
translation = transformation_matrix[:3, 3]

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

scale = cv2.norm(unrectify_params['opt_translation'][:2]) / cv2.norm(translation[:2])
translation *= scale

print(Rotation.from_matrix(rotation_matrix).as_euler('zyx', degrees=True))
print(translation)

image_size = tuple(unrectify_params['image_size'])

left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']

left_rotation, right_rotation, left_projection, right_projection, Q, left_roi, right_roi = cv2.stereoRectify(left_camera_matrix, np.zeros(5),
                                                                                                             right_camera_matrix, np.zeros(5),
                                                                                                             image_size, rotation_matrix, translation)

left_mapx, left_mapy = cv2.initUndistortRectifyMap(left_camera_matrix, np.zeros(5), left_rotation, left_projection, image_size, cv2.CV_32F)
right_mapx, right_mapy = cv2.initUndistortRectifyMap(right_camera_matrix, np.zeros(5), right_rotation, right_projection, image_size, cv2.CV_32F)

np.savez(config.calib_dir + config.calib_sub_dir + 'stereo_params',
         rotation_matrix=rotation_matrix,
         translation=translation,
         left_rotation=left_rotation,
         right_rotation=right_rotation,
         left_projection=left_projection,
         right_projection=right_projection,
         Q=Q,
         left_mapx=left_mapx,
         left_mapy=left_mapy,
         right_mapx=right_mapx,
         right_mapy=right_mapy,
         left_roi=left_roi,
         right_roi=right_roi)
