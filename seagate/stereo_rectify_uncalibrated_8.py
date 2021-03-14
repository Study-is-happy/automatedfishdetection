import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from MultiHarrisZernike import MultiHarrisZernike

import config
import utils

left_image = cv2.imread(config.calib_dir + 'port.png')
right_image = cv2.imread(config.calib_dir + 'stbd.png')

gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

left_image_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_image_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

image_size = (gray_left_image.shape[1], gray_left_image.shape[0])

# sift_30000 = cv2.xfeatures2d.SIFT_create(nfeatures=30000)
zernike_10000 = MultiHarrisZernike(Nfeats=10000, seci=5, secj=4, levels=6, ratio=0.75,
                                   sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)
zernike_30000 = MultiHarrisZernike(Nfeats=30000, seci=5, secj=4, levels=6, ratio=0.75,
                                   sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)
zernike_60000 = MultiHarrisZernike(Nfeats=60000, seci=5, secj=4, levels=6, ratio=0.75,
                                   sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

left_keypoints, left_descriptors = zernike_30000.detectAndCompute(gray_left_image, np.uint8(left_image_mask != 0))
right_keypoints, right_descriptors = zernike_30000.detectAndCompute(gray_right_image, np.uint8(right_image_mask != 0))

left_non_max_suppression_mask = utils.get_non_max_suppression_mask(left_keypoints, gray_left_image.shape)

left_keypoints = np.array(left_keypoints)[left_non_max_suppression_mask]
left_descriptors = np.array(left_descriptors)[left_non_max_suppression_mask]

right_non_max_suppression_mask = utils.get_non_max_suppression_mask(right_keypoints, gray_right_image.shape)

right_keypoints = np.array(right_keypoints)[right_non_max_suppression_mask]
right_descriptors = np.array(right_descriptors)[right_non_max_suppression_mask]

matcher = cv2.BFMatcher(cv2.NORM_L2)

matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)

left_points = []
right_points = []

for match_1, match_2 in matches:
    if match_1.distance < 0.75 * match_2.distance:
        left_points.append(left_keypoints[match_1.queryIdx].pt)
        right_points.append(right_keypoints[match_1.trainIdx].pt)

left_points = np.array(left_points)
right_points = np.array(right_points)

fundamental_matrix, fundamental_mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_RANSAC, 0.1)

fundamental_mask = (fundamental_mask.ravel() == 1)
left_points = left_points[fundamental_mask]
right_points = right_points[fundamental_mask]

print(len(left_points))

# utils.plot_match_points(left_image, right_image, left_points, right_points)

# _, left_homography_matrix, right_homography_matrix = cv2.stereoRectifyUncalibrated(left_points,
#                                                                                    right_points,
#                                                                                    fundamental_matrix, image_size)

# left_image = cv2.warpPerspective(left_image, left_homography_matrix, image_size)

# right_image = cv2.warpPerspective(right_image, right_homography_matrix, image_size)

# plt.imshow(left_image)
# plt.show()
# plt.imshow(right_image)
# plt.show()

left_camera_matrix = np.array([[3200.30144, 0, 1238.67489],
                               [0, 3194.1071, 991.12046],
                               [0, 0, 1]])

right_camera_matrix = np.array([[3210.14443, 0, 1238.01240],
                                [0, 3204.52724, 1009.88894],
                                [0, 0, 1]])

norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((left_points.T, np.ones(len(left_points)))))[:-1].T
norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((right_points.T, np.ones(len(right_points)))))[:-1].T

essential_matrix = right_camera_matrix.T.dot(fundamental_matrix).dot(left_camera_matrix)

_, rotation, translation, _ = cv2.recoverPose(essential_matrix, norm_left_points, norm_right_points, np.eye(3))

print(Rotation.from_matrix(rotation).as_euler('zyx', degrees=True))
print(translation)
