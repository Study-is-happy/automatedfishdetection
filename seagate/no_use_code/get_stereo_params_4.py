import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

import config
import seagate_utils

match_points_dir = config.calib_dir + config.calib_sub_dir + 'all_match_points/'

all_left_points = []
all_right_points = []

test_left_points_list = []
test_right_points_list = []
test_match_points_files = []

for match_points_file in os.listdir(match_points_dir):

    match_points = np.load(match_points_dir + match_points_file)

    left_points = match_points['left_points']
    right_points = match_points['right_points']

    all_left_points.extend(left_points)
    all_right_points.extend(right_points)

    test_left_points_list.append(left_points)
    test_right_points_list.append(right_points)
    test_match_points_files.append(match_points_file)

all_left_points = np.array(all_left_points)
all_right_points = np.array(all_right_points)

_, homography_mask = cv2.findHomography(all_left_points, all_right_points, cv2.RANSAC, 3.0)

homography_mask = (homography_mask.ravel() == 0)
all_left_points = all_left_points[homography_mask]
all_right_points = all_right_points[homography_mask]

points_len = len(all_left_points)

print(points_len)

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']

norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((all_left_points.T, np.ones(points_len))))[:-1].T
norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((all_right_points.T, np.ones(points_len))))[:-1].T

iteration = 0
best_inlier_count = 0

while iteration < 100 or best_inlier_count < points_len / 2:

    random_indexes = random.sample(range(points_len), 8)
    random_left_points = norm_left_points[random_indexes]
    random_right_points = norm_right_points[random_indexes]

    essential_matrix, _ = cv2.findEssentialMat(random_left_points, random_right_points, np.eye(3))

    _, rotation_matrix, translation, _ = cv2.recoverPose(essential_matrix, random_left_points, random_right_points, np.eye(3))

    # rotation_matrix = np.linalg.inv(rotation_matrix)
    # translation = -rotation_matrix.dot(translation)

    left_projection_matrix = left_camera_matrix.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    right_projection_matrix = right_camera_matrix.dot(np.hstack((rotation_matrix, translation)))

    object_points = cv2.triangulatePoints(
        left_projection_matrix, right_projection_matrix, all_left_points.T, all_right_points.T)
    object_points = (object_points / object_points[-1])[:-1].T

    left_reproject_points, _ = cv2.projectPoints(object_points, np.zeros(3), np.zeros((3, 1)), left_camera_matrix, np.zeros(5))
    right_reproject_points, _ = cv2.projectPoints(object_points, cv2.Rodrigues(rotation_matrix)[0], translation, right_camera_matrix, np.zeros(5))
    left_reproject_points = np.squeeze(left_reproject_points)
    right_reproject_points = np.squeeze(right_reproject_points)
    left_reproject_errors = np.sum((all_left_points - left_reproject_points)**2, axis=1)
    right_reproject_errors = np.sum((all_right_points - right_reproject_points)**2, axis=1)

    inlier_mask = (left_reproject_errors < 1) & (right_reproject_errors < 1)
    inlier_count = np.count_nonzero(inlier_mask)

    if inlier_count > best_inlier_count:
        best_inlier_count = inlier_count
        best_inlier_mask = inlier_mask
        best_rotation_matrix = rotation_matrix
        best_translation = translation
        print(best_inlier_count)
        print(object_points)

    iteration += 1

all_left_points = all_left_points[best_inlier_mask]
all_right_points = all_right_points[best_inlier_mask]

print(len(all_left_points))

np.savez(config.calib_dir + config.calib_sub_dir + 'match_points',
         left_camera_matrix=left_camera_matrix,
         right_camera_matrix=right_camera_matrix,
         left_points=all_left_points,
         right_points=all_right_points,
         rotation_matrix=best_rotation_matrix,
         translation=best_translation
         )

outlier_left_points = np.array([[0, 1000]])
outlier_right_points = np.array([[1000, 1000]])
left_projection_matrix = left_camera_matrix.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
right_projection_matrix = right_camera_matrix.dot(np.hstack((best_rotation_matrix, best_translation)))
object_points = cv2.triangulatePoints(
    left_projection_matrix, right_projection_matrix, outlier_left_points.T, outlier_right_points.T)
object_points = (object_points / object_points[-1])[:-1].T
left_reproject_points, _ = cv2.projectPoints(object_points, np.zeros(3), np.zeros((3, 1)), left_camera_matrix, np.zeros(5))
right_reproject_points, _ = cv2.projectPoints(object_points, cv2.Rodrigues(rotation_matrix)[0], translation, right_camera_matrix, np.zeros(5))
print(left_reproject_points)
print(right_reproject_points)

test_mask = homography_mask.copy()
test_mask_indexes = np.where(test_mask)[0]
test_mask[test_mask_indexes] = best_inlier_mask

test_seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-19-02/d20191102_17/'
test_left_dir = test_seagate_dir + 'port/port_rectified/'
test_right_dir = test_seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

current_index = 0
for test_match_points_file, test_left_points, test_right_points in zip(test_match_points_files, test_left_points_list, test_right_points_list):

    next_index = current_index + len(test_left_points)

    test_match_points_file = os.path.splitext(test_match_points_file)[0]
    left_image_id = test_match_points_file[:32]
    right_image_id = test_match_points_file[33:]

    current_mask = test_mask[current_index:next_index]
    test_left_points = test_left_points[current_mask]
    test_right_points = test_right_points[current_mask]

    left_image = cv2.imread(test_left_dir + left_image_id + '.tif')
    right_image = cv2.imread(test_right_dir + right_image_id + '.tif')
    print(test_left_dir + left_image_id + '.tif')
    print(test_right_dir + right_image_id + '.tif')

    left_image[left_mask == 0] = 0
    right_image[right_mask == 0] = 0

    left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
    right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

    seagate_utils.plot_match_points(left_image, right_image, test_left_points, test_right_points)

    current_index = next_index
