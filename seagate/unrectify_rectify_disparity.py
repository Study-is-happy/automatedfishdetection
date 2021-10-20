import numpy as np
import cv2
import matplotlib.pyplot as plt

import config

left_image_id = '20191102.175853.02722_rect_color'
right_image_id = '20191102.175853.02721_rect_color'

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
stereo_params = np.load(config.calib_dir + config.calib_sub_dir + 'stereo_params/' + left_image_id + '=' + right_image_id + '.npz')
# stereo_params = np.load(config.calib_dir + config.calib_sub_dir + 'stereo_params.npz')

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-19-02/d20191102_17/'
left_dir = seagate_dir + 'port/port_rectified/'
right_dir = seagate_dir + 'stbd/stbd_rectified/'

left_image = cv2.imread(left_dir + left_image_id + '.tif')
right_image = cv2.imread(right_dir + right_image_id + '.tif')

left_image[left_mask == 0] = 0
right_image[right_mask == 0] = 0

plt.subplot(121)
plt.imshow(left_image)
plt.subplot(122)
plt.imshow(right_image)
plt.show()

left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

left_image = cv2.remap(left_image, stereo_params['left_mapx'], stereo_params['left_mapy'], cv2.INTER_LINEAR)
right_image = cv2.remap(right_image, stereo_params['right_mapx'], stereo_params['right_mapy'], cv2.INTER_LINEAR)

stereo_disparity = cv2.StereoSGBM_create(minDisparity=293,
                                         numDisparities=16 * 20,
                                         blockSize=5,
                                         P1=8 * 3 * 5**2,
                                         P2=32 * 3 * 5**2,
                                         # disp12MaxDiff=0,
                                         # preFilterCap=32,
                                         # uniquenessRatio=15,
                                         mode=cv2.STEREO_SGBM_MODE_HH
                                         )

disparity_map = np.float32(stereo_disparity.compute(left_image, right_image)) / 16

disparity_image = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disparity_image = cv2.cvtColor(disparity_image, cv2.COLOR_GRAY2RGB)

for row in range(0, 2050 + 41, 41):
    left_image = cv2.line(left_image, (0, row), (2448, row), (0, 255, 0), 2)
    right_image = cv2.line(right_image, (0, row), (2448, row), (0, 255, 0), 2)

plt.subplot(121)
plt.imshow(left_image)
plt.subplot(122)
plt.imshow(disparity_image)
plt.show()
