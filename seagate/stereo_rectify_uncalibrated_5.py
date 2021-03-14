import numpy as np
import cv2
import matplotlib.pyplot as plt
from MultiHarrisZernike import MultiHarrisZernike

import utils

left_image = cv2.imread('/home/auv/venv3/port_2.png')
right_image = cv2.imread('/home/auv/venv3/stbd_2.png')

gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

left_image_mask = cv2.imread('/home/auv/venv3/port_2_mask.png', -1)
right_image_mask = cv2.imread('/home/auv/venv3/stbd_mask.png', -1)

image_size = (gray_left_image.shape[1], gray_left_image.shape[0])

# zernike_obj = MultiHarrisZernike(Nfeats=30000, seci=5, secj=4, levels=6, ratio=0.75,
#                                  sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

# left_keypoints, left_descriptors = zernike_obj.detectAndCompute(gray_left_image, mask=left_image_mask)
# right_keypoints, right_descriptors = zernike_obj.detectAndCompute(gray_right_image, mask=right_image_mask)

sift = cv2.xfeatures2d.SIFT_create()

left_keypoints, left_descriptors = sift.detectAndCompute(gray_left_image, left_image_mask)
right_keypoints, right_descriptors = sift.detectAndCompute(gray_right_image, right_image_mask)

left_non_max_suppression_mask = utils.get_non_max_suppression_mask(left_keypoints, gray_left_image.shape)

left_keypoints = np.array(left_keypoints)[left_non_max_suppression_mask]
left_descriptors = np.array(left_descriptors)[left_non_max_suppression_mask]

right_non_max_suppression_mask = utils.get_non_max_suppression_mask(right_keypoints, gray_right_image.shape)

right_keypoints = np.array(right_keypoints)[right_non_max_suppression_mask]
right_descriptors = np.array(right_descriptors)[right_non_max_suppression_mask]

# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
# matcher = cv2.FlannBasedMatcher(index_params, search_params)

matcher = cv2.BFMatcher(cv2.NORM_L2)

matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
cross_matches = matcher.match(right_descriptors, left_descriptors)

cross_match_dict = {}
for cross_match in cross_matches:
    cross_match_dict[cross_match.trainIdx] = cross_match.queryIdx

left_points = []
right_points = []

for match_1, match_2 in matches:
    # if match_1.queryIdx in cross_match_dict and cross_match_dict[match_1.queryIdx] == match_1.trainIdx and match_1.distance < 0.75 * match_2.distance:
    if match_1.distance < 0.75 * match_2.distance:
        left_points.append(left_keypoints[match_1.queryIdx].pt)
        right_points.append(right_keypoints[match_1.trainIdx].pt)

left_points = np.array(left_points)
right_points = np.array(right_points)

_, homography_mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, 3.0)

homography_mask = (homography_mask.ravel() == 0)
left_points = left_points[homography_mask]
right_points = right_points[homography_mask]

utils.plot_match_points(left_image, right_image, left_points, right_points)

fundamental_mat, fundamental_mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_RANSAC, 0.1)

print(fundamental_mat)

fundamental_mask = (fundamental_mask.ravel() == 1)
left_points = left_points[fundamental_mask]
right_points = right_points[fundamental_mask]

utils.plot_match_points(left_image, right_image, left_points, right_points)

_, left_homography_matrix, right_homography_matrix = cv2.stereoRectifyUncalibrated(left_points,
                                                                                   right_points,
                                                                                   fundamental_mat, image_size)

left_image = cv2.warpPerspective(left_image, np.array([[1., 0., 32.],
                                                       [0., 1., 0.],
                                                       [0., 0., 1.]]).dot(left_homography_matrix), image_size)
right_image = cv2.warpPerspective(right_image, right_homography_matrix, image_size)

disparity_mask = cv2.warpPerspective(left_image_mask, left_homography_matrix, image_size)

# cv2.imwrite('/home/auv/venv3/port_2_my_rect.png', left_image)
# cv2.imwrite('/home/auv/venv3/stbd_2_my_rect.png', right_image)
# cv2.imwrite('/home/auv/venv3/port_2_disparity_mask.png', disparity_mask)
plt.imshow(left_image)
plt.show()
plt.imshow(right_image)
plt.show()
plt.imshow(disparity_mask)
plt.show()
