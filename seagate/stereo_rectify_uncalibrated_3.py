from MultiHarrisZernike import MultiHarrisZernike
import numpy as np
import cv2
import matplotlib.pyplot as plt

import util

left_image = cv2.imread('/home/auv/venv3/port_2.png')
right_image = cv2.imread('/home/auv/venv3/stbd_2.png')

gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

left_image_mask = cv2.imread('/home/auv/venv3/port_mask.png', -1)
right_image_mask = cv2.imread('/home/auv/venv3/stbd_mask.png', -1)

image_size = (gray_left_image.shape[1], gray_left_image.shape[0])

zernike_obj = MultiHarrisZernike(Nfeats=30000, seci=5, secj=4, levels=6, ratio=0.75,
                                 sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

left_keypoints, left_descriptors = zernike_obj.detectAndCompute(gray_left_image, mask=left_image_mask)
right_keypoints, right_descriptors = zernike_obj.detectAndCompute(gray_right_image, mask=right_image_mask)

bf_matcher = cv2.BFMatcher()
matches = bf_matcher.knnMatch(left_descriptors, right_descriptors, k=2)

# sift = cv2.xfeatures2d.SIFT_create()

# left_keypoints, left_descriptors = sift.detectAndCompute(gray_left_image, left_image_mask)
# right_keypoints, right_descriptors = sift.detectAndCompute(gray_right_image, right_image_mask)

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)

# flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann_matcher.knnMatch(left_descriptors, right_descriptors, k=2)


left_points = []
right_points = []

for match_1, match_2 in matches:
    if match_1.distance < 0.75 * match_2.distance:
        left_points.append(left_keypoints[match_1.queryIdx].pt)
        right_points.append(right_keypoints[match_1.trainIdx].pt)

left_points = np.array(left_points)
right_points = np.array(right_points)


fundamental_mat, good_mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_RANSAC, 0.1)
print(fundamental_mat)

good_mask = good_mask.ravel() == 1
print(len(good_mask))
print(np.count_nonzero(good_mask))

left_points = left_points[good_mask]
right_points = right_points[good_mask]

util.plot_match_points(left_image, right_image, left_points, right_points)

left_camera_matrix = np.array([[3200.30144, 0, 1238.67489],
                               [0, 3194.1071, 991.12046],
                               [0, 0, 1]])

right_camera_matrix = np.array([[3210.14443, 0, 1238.01240],
                                [0, 3204.52724, 1009.88894],
                                [0, 0, 1]])

essential_matrix = right_camera_matrix.T.dot(fundamental_mat).dot(left_camera_matrix)
left_rotation, right_rotation, translation = cv2.decomposeEssentialMat(essential_matrix)

print(cv2.Rodrigues(left_rotation))
print(cv2.Rodrigues(right_rotation))

left_image = cv2.warpPerspective(left_image, left_camera_matrix.dot(left_rotation), image_size)
right_image = cv2.warpPerspective(right_image, right_rotation.dot(np.linalg.inv(right_camera_matrix)), image_size)

# disparity_mask = cv2.warpPerspective(left_image_mask, right_camera_matrix.dot(R1), image_size)

# cv2.imwrite('/home/auv/venv3/port_2_my_rect.png', left_image)
# cv2.imwrite('/home/auv/venv3/stbd_2_my_rect.png', right_image)
# cv2.imwrite('/home/auv/venv3/port_2_disparity_mask.png', disparity_mask)
plt.imshow(left_image)
plt.show()
plt.imshow(right_image)
plt.show()
# plt.imshow(disparity_mask)
# plt.show()
