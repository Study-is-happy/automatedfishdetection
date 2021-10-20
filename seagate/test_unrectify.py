import numpy as np
import cv2

import config

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

left_mask = cv2.remap(left_mask, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_NEAREST)
right_mask = cv2.remap(right_mask, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_NEAREST)

cv2.imwrite(config.calib_dir + 'SH-17-09_9_10_unrectify_left_mask.png', left_mask)
cv2.imwrite(config.calib_dir + 'SH-17-09_9_10_unrectify_right_mask.png', right_mask)
