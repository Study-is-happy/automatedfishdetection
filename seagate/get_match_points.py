import numpy as np
import cv2
import glob
import shutil
import os
import matplotlib.pyplot as plt

# from MultiHarrisZernike import MultiHarrisZernike
from My_detector_3 import My_detector

import config
import seagate_utils

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

left_remap_mask = cv2.remap(left_mask, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_NEAREST)
right_remap_mask = cv2.remap(right_mask, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_NEAREST)

# zernike = MultiHarrisZernike(Nfeats=41 * 48 * 20, seci=41, secj=48, levels=12, ratio=0.75,
#                              sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

sift = cv2.SIFT_create()
num_octaves = 11

# my_detector = My_detector()

for sub_dir in os.listdir(config.seagate_dir):

    sub_dir += '/'

    seagate_sub_dir = config.seagate_dir + sub_dir

    if os.path.isdir(seagate_sub_dir):

        calib_sub_dir = config.calib_dir + sub_dir

        print(calib_sub_dir)

        if not os.path.exists(calib_sub_dir):

            #  shutil.rmtree(calib_sub_dir)
            os.mkdir(calib_sub_dir)

            left_dir = seagate_sub_dir + 'port/port_rectified/'
            right_dir = seagate_sub_dir + 'stbd/stbd_rectified/'

            match_points_dir = calib_sub_dir + 'match_points/'

            os.mkdir(match_points_dir)

            for left_image_path, right_image_path in zip(sorted(glob.glob(left_dir + '*.tif') + glob.glob(left_dir + '*.jpg')),
                                                         sorted(glob.glob(right_dir + '*.tif') + glob.glob(right_dir + '*.jpg'))):

                # if left_image_path != left_dir + '20161027.175930.00463_rect_color.jpg':
                #     continue

                print(left_image_path)

                left_image_id = os.path.splitext(os.path.basename(left_image_path))[0]
                right_image_id = os.path.splitext(os.path.basename(right_image_path))[0]

                left_image = cv2.imread(left_image_path)
                right_image = cv2.imread(right_image_path)

                left_image[left_mask == 0] = 0
                right_image[right_mask == 0] = 0

                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

                left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
                right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

                # left_keypoints_list, left_descriptors_list = my_detector.detectAndCompute(left_image, left_remap_mask)
                # right_keypoints_list, right_descriptors_list = my_detector.detectAndCompute(right_image, right_remap_mask)

                left_keypoints, left_descriptors = sift.detectAndCompute(left_image, left_remap_mask)
                right_keypoints, right_descriptors = sift.detectAndCompute(right_image, right_remap_mask)

                left_keypoints_mask = seagate_utils.get_non_max_suppression_mask(left_keypoints, left_image.shape)
                right_keypoints_mask = seagate_utils.get_non_max_suppression_mask(right_keypoints, right_image.shape)

                left_keypoints = np.array(left_keypoints)[left_keypoints_mask]
                left_descriptors = np.array(left_descriptors)[left_keypoints_mask]
                right_keypoints = np.array(right_keypoints)[right_keypoints_mask]
                right_descriptors = np.array(right_descriptors)[right_keypoints_mask]

                left_keypoints_list, left_descriptors_list = seagate_utils.keypoints_2_keypoints_list(left_keypoints, left_descriptors, num_octaves)
                right_keypoints_list, right_descriptors_list = seagate_utils.keypoints_2_keypoints_list(right_keypoints, right_descriptors, num_octaves)

                left_points, right_points = seagate_utils.multi_scale_match(left_keypoints_list, right_keypoints_list, left_descriptors_list, right_descriptors_list, left_image, right_image)

                print(len(left_points))

                # if len(left_points) > 1000:

                # seagate_utils.plot_match_points(left_image, right_image, left_points, right_points, False)

                np.savez(match_points_dir + left_image_id + '=' + right_image_id,
                         left_points=left_points,
                         right_points=right_points)
