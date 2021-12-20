import numpy as np
import cv2
import json
import glob
import shutil
import os
import matplotlib.pyplot as plt

from MultiHarrisZernike import MultiHarrisZernike

import config
import utils
import seagate_utils

# for calib_dir in ['RL-16_06', 'RL-19-02', 'SH-17-09', 'SH-18-12']:
#     for calib_sub_dir in os.listdir(seagate_dir + calib_dir):
#         if os.path.isdir(calib_sub_dir):
#             print(calib_sub_dir)

seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-19-02/d20191102_17/'

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']

left_dir = seagate_dir + 'port/port_rectified/'
right_dir = seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

image_shape = np.flip(unrectify_params['image_size'])

left_roi = seagate_utils.get_roi_mask(image_shape, unrectify_params['left_roi'])
right_roi = seagate_utils.get_roi_mask(image_shape, unrectify_params['right_roi'])

zernike = MultiHarrisZernike(Nfeats=10000, seci=5, secj=4, levels=6, ratio=0.75,
                             sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

# sift = cv2.SIFT_create()

with open(config.project_dir + 'all/instances_all.json') as instances_file:
    instances_dict = json.load(instances_file)

match_points_dir = config.calib_dir + config.calib_sub_dir + 'all_match_points/'
# if os.path.exists(match_points_dir):
#     shutil.rmtree(match_points_dir)
# os.mkdir(match_points_dir)

for left_image_path, right_image_path in zip(sorted(glob.glob(left_dir + '*.tif')),
                                             sorted(glob.glob(right_dir + '*.tif'))):

    # if left_image_path != left_dir + '20191102.160147.00088_rect_color.tif':
    #     continue

    # if left_image_path != left_dir + '20191102.175909.02728_rect_color.tif':
    #     continue

    print(left_image_path)
    print(right_image_path)

    left_image_id = os.path.splitext(os.path.basename(left_image_path))[0]
    right_image_id = os.path.splitext(os.path.basename(right_image_path))[0]

    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    left_image[left_mask == 0] = 0
    right_image[right_mask == 0] = 0

    left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
    right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

    gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    left_points, right_points = seagate_utils.get_match_points(zernike, gray_left_image, gray_right_image, left_roi, right_roi)

    if len(left_points) < 8:
        continue

    # norm_left_points = np.linalg.inv(left_camera_matrix).dot(np.vstack((left_points.T, np.ones(len(left_points)))))[:-1].T
    # norm_right_points = np.linalg.inv(right_camera_matrix).dot(np.vstack((right_points.T, np.ones(len(right_points)))))[:-1].T

    # _, essential_mask = cv2.findEssentialMat(norm_left_points, norm_right_points, np.eye(3), cv2.RANSAC, 0.999, 0.00005)
    # _, homography_mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, 1.0)

    # essential_mask_count = np.count_nonzero(essential_mask)
    # homography_mask_count = np.count_nonzero(homography_mask)

    # print(essential_mask_count, homography_mask_count)

    # if essential_mask_count == 0 and homography_mask_count == 0:
    #     continue

    # if essential_mask_count > homography_mask_count:
    #     essential_mask = (essential_mask.ravel() == 1)
    #     left_points = left_points[essential_mask]
    #     right_points = right_points[essential_mask]

    # else:
    #     homography_mask = (homography_mask.ravel() == 1)
    #     left_points = left_points[homography_mask]
    #     right_points = right_points[homography_mask]

    print(len(left_points))
    if len(left_points) > 200:
        # seagate_utils.plot_match_points(left_image, right_image, left_points, right_points)
        np.savez(match_points_dir + left_image_id + '=' + right_image_id,
                 left_points=left_points,
                 right_points=right_points)
