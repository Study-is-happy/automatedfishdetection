import numpy as np
import cv2
import matplotlib.pyplot as plt

import config

left_image = cv2.imread(config.calib_dir + config.calib_sub_dir + 'port_my_rect.png')
right_image = cv2.imread(config.calib_dir + config.calib_sub_dir + 'stbd_my_rect.png')

height, width, _ = left_image.shape

for row in range(0, height, 50):
    left_image = cv2.line(left_image, (0, row), (width, row), (0, 255, 0), 2)
    right_image = cv2.line(right_image, (0, row), (width, row), (0, 255, 0), 2)

for col in range(0, width, 48):
    left_image = cv2.line(left_image, (col, 0), (col, height), (0, 255, 0), 2)
    right_image = cv2.line(right_image, (col, 0), (col, height), (0, 255, 0), 2)

plt.subplot(121)
plt.imshow(left_image)
plt.subplot(122)
plt.imshow(right_image)
plt.show()

# img2 = cv2.warpAffine(img2, rotation_matrix, (width, height))
# img2 = cv2.warpAffine(img2, translation_matrix, (width, height))

stereo_disparity = cv2.StereoSGBM_create(minDisparity=150,
                                         numDisparities=128,
                                         blockSize=5,
                                         P1=8 * 3 * 5**2,
                                         P2=32 * 3 * 5**2,
                                         disp12MaxDiff=1,
                                         preFilterCap=32,
                                         uniquenessRatio=15,
                                         # speckleWindowSize=100,
                                         # speckleRange=32,
                                         mode=cv2.STEREO_SGBM_MODE_HH)

disparity_map = np.float32(stereo.compute(left_image, right_image)) / 16

plt.imshow(disparity_map, 'gray')
plt.show()

# plt.hist(disparity_map.flatten(), bins='auto')
# plt.show()

# disparity_image = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# disparity_image = cv2.cvtColor(disparity_image, cv2.COLOR_GRAY2RGB)
# cv2.imwrite(config.calib_dir + config.calib_sub_dir + 'vis_disparity_map.png', disparity_image)
