import numpy as np
import cv2
import matplotlib.pyplot as plt

import config

color_img1 = cv2.imread(config.calib_dir + 'port_my_rect.png')
color_img2 = cv2.imread(config.calib_dir + 'stbd_my_rect.png')

height, width, _ = color_img1.shape

for i in range(0, height, 50):
    color_img1 = cv2.line(color_img1, (0, i), (width, i), (0, 255, 0), 2)
    color_img2 = cv2.line(color_img2, (0, i), (width, i), (0, 255, 0), 2)

for i in range(0, width, 48):
    color_img1 = cv2.line(color_img1, (i, 0), (i, height), (0, 255, 0), 2)
    color_img2 = cv2.line(color_img2, (i, 0), (i, height), (0, 255, 0), 2)

# plt.subplot(121)
# plt.imshow(color_img1)
# plt.subplot(122)
# plt.imshow(color_img2)
# plt.show()


img1 = cv2.imread(config.calib_dir + 'port_my_rect.png', 0)
img2 = cv2.imread(config.calib_dir + 'stbd_my_rect.png', 0)

# img2 = cv2.warpAffine(img2, rotation_matrix, (width, height))
# img2 = cv2.warpAffine(img2, translation_matrix, (width, height))

stereo = cv2.StereoSGBM_create(minDisparity=-160,
                               numDisparities=320,
                               blockSize=5,
                               uniquenessRatio=5,
                               speckleWindowSize=0,
                               speckleRange=1,
                               disp12MaxDiff=0,
                               P1=8 * 3 * 5**2,
                               P2=32 * 3 * 5**2)

disparity_map = np.float32(stereo.compute(img1, img2)) / 16

disparity_mask = cv2.imread(config.calib_dir + 'disparity_mask.png', -1)

disparity_map[disparity_mask == 0] = np.min(disparity_map)

# plt.hist(disparity_map.flatten(), bins='auto')
# plt.show()

disparity_map += 10

disparity_map[disparity_map < 0] = 0

disparity_map = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

disparity_map = cv2.cvtColor(disparity_map, cv2.COLOR_GRAY2BGR)
disparity_mask[disparity_mask != 255] = 0

disparity_mask, contours, _ = cv2.findContours(disparity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(disparity_map, contours, -1, (0, 255, 0), 3)

plt.imshow(disparity_map, 'gray')
plt.show()

# Q = np.array([[1., 0., 0., -913.74072],
#               [0., 1., 0., -999.57969],
#               [0., 0., 0., 3194.1071],
#               [0., 0., 0.00549, 0.26763]])

# depth_map = cv2.reprojectImageTo3D(disparity_map, Q)

# plt.imshow(depth_map[:, :, 2], 'gray')
# plt.show()
