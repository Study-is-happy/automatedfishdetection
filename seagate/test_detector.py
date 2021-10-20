import cv2
import matplotlib.pyplot as plt

from MultiHarrisZernike import MultiHarrisZernike
from My_matcher import My_matcher

import config
import seagate_utils

import numpy as np
print(np.sqrt(15**2 + 15**2))

zernike = MultiHarrisZernike(Nfeats=41 * 48 * 10, seci=41, secj=48, levels=12, ratio=0.75,
                             sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

my_matcher = My_matcher()

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

left_image = cv2.imread('/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-19-02/d20191102_17/port/port_rectified/20191102.160139.00085_rect_color.tif', 0)
right_image = cv2.imread('/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-19-02/d20191102_17/stbd/stbd_rectified/20191102.160139.00084_rect_color.tif', 0)

left_image[left_mask == 0] = 0
right_image[right_mask == 0] = 0

left_points, right_points = my_matcher.match(left_image, right_image, left_mask, right_mask)

# plt.imshow(cv2.drawKeypoints(image, keypoints, None, (0, 255, 0)))
# plt.show()

seagate_utils.plot_match_points(left_image, right_image, left_points, right_points, False)
