from MultiHarrisZernike import MultiHarrisZernike
import numpy as np
import cv2
import matplotlib.pyplot as plt

import util

left_image = cv2.imread('/home/auv/venv3/port_2.png')
right_image = cv2.imread('/home/auv/venv3/stbd_2.png')

gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

left_image_mask = cv2.imread('/home/auv/venv3/port_2_mask.png', -1)
right_image_mask = cv2.imread('/home/auv/venv3/stbd_mask.png', -1)

image_size = (gray_left_image.shape[1], gray_left_image.shape[0])

left_zernike_obj = MultiHarrisZernike(Nfeats=1000, seci=5, secj=4, levels=12, ratio=0.75,
                                      sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)
left_keypoints, left_descriptors = left_zernike_obj.detectAndCompute(gray_left_image, mask=left_image_mask)

right_zernike_obj = MultiHarrisZernike(Nfeats=30000, seci=5, secj=4, levels=12, ratio=0.75,
                                       sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)
right_keypoints, right_descriptors = right_zernike_obj.detectAndCompute(gray_right_image, mask=right_image_mask)

bf_matcher = cv2.BFMatcher()
matches = bf_matcher.knnMatch(left_descriptors, right_descriptors, k=2)

# sift = cv2.xfeatures2d.SIFT_create()

# left_keypoints, left_descriptors = sift.detectAndCompute(gray_left_image, left_image_mask)
# right_keypoints, right_descriptors = sift.detectAndCompute(gray_right_image, right_image_mask)

# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)

# flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann_matcher.knnMatch(left_descriptors, right_descriptors, k=2)

plot_left_keypoints = left_zernike_obj.detect(gray_left_image, mask=left_image_mask)
plt.imshow(cv2.drawKeypoints(left_image, plot_left_keypoints, None, color=(0, 255, 0)))
plt.show()


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


def get_homo_point(point):
    return np.array([point[0], point[1], 1])


good_mask = []
for left_point, right_point in zip(left_points, right_points):
    good_mask.append(np.abs(get_homo_point(right_point).dot(fundamental_mat).dot(get_homo_point(left_point).T)) < 0.01)

print(np.count_nonzero(good_mask))

left_points = left_points[good_mask]
right_points = right_points[good_mask]

# random_mask = np.random.choice(len(left_points), 300)
# left_points = left_points[random_mask]
# right_points = right_points[random_mask]

_, left_homography_matrix, right_homography_matrix = cv2.stereoRectifyUncalibrated(left_points,
                                                                                   right_points,
                                                                                   fundamental_mat, image_size)

# left_homography_matrix = np.array([[-1.68959100e-02, 1.50560901e-03, 1.00996607e+00],
#                                    [-1.98659502e-03, -1.66002665e-02, 2.51742597e+00],
#                                    [-3.04643426e-07, 3.85971654e-08, -1.63304407e-02]])
# right_homography_matrix = np.array([[1.02786676e+00, -5.30021490e-02, 2.02182852e+01],
#                                     [7.59440317e-02, 9.97412533e-01, -9.03033414e+01],
#                                     [2.38509819e-05, -1.22988051e-06, 9.72067026e-01]])

left_image = cv2.warpPerspective(left_image, np.array([[1., 0., 0.],
                                                       [0., 1., 0.],
                                                       [0., 0., 1.]]).dot(left_homography_matrix), image_size)
right_image = cv2.warpPerspective(right_image, right_homography_matrix, image_size)

disparity_mask = cv2.warpPerspective(left_image_mask, left_homography_matrix, image_size)

cv2.imwrite('/home/auv/venv3/port_2_my_rect.png', left_image)
cv2.imwrite('/home/auv/venv3/stbd_2_my_rect.png', right_image)
cv2.imwrite('/home/auv/venv3/port_2_disparity_mask.png', disparity_mask)
plt.imshow(left_image)
plt.show()
plt.imshow(right_image)
plt.show()
plt.imshow(disparity_mask)
plt.show()
