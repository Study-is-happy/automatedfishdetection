import numpy as np
import cv2
import glob
import shutil
import os
import matplotlib.pyplot as plt

from MultiHarrisZernike import MultiHarrisZernike
from My_detector_3 import My_detector

import config
import utils
import seagate_utils

# for calib_dir in ['RL-16_06', 'RL-19-02', 'SH-17-09', 'SH-18-12']:
#     for calib_sub_dir in os.listdir(seagate_dir + calib_dir):
#         if os.path.isdir(calib_sub_dir):
#             print(calib_sub_dir)

seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/SH-18-12/d20181015_3/'

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

left_dir = seagate_dir + 'port/port_rectified/'
right_dir = seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

left_remap_mask = cv2.remap(left_mask, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_NEAREST)
right_remap_mask = cv2.remap(right_mask, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_NEAREST)

# zernike = MultiHarrisZernike(Nfeats=41 * 48 * 20, seci=41, secj=48, levels=12, ratio=0.75,
#                              sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

# sift = cv2.SIFT_create(nOctaveLayers=12, contrastThreshold=0.04, sigma=1.5)

my_detector = My_detector()

match_points_dir = config.calib_dir + config.calib_sub_dir + 'all_match_points/'
if os.path.exists(match_points_dir):
    shutil.rmtree(match_points_dir)
os.mkdir(match_points_dir)

for left_image_path, right_image_path in zip(sorted(glob.glob(left_dir + '*.tif')),
                                             sorted(glob.glob(right_dir + '*.tif'))):

    # if left_image_path != left_dir + '20191102.160747.00223_rect_color.tif':
    #     continue

    # if left_image_path != left_dir + '20191102.175901.02725_rect_color.tif':
    #     continue

    # if left_image_path == left_dir + '20191102.160131.00082_rect_color.tif':
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

    left_keypoints_list, left_descriptors_list = my_detector.detectAndCompute(left_image, left_remap_mask)
    right_keypoints_list, right_descriptors_list = my_detector.detectAndCompute(right_image, right_remap_mask)

    # plt.subplot(121)
    # plt.imshow(cv2.drawKeypoints(left_image, left_keypoints_list[0], None, (0, 255, 0)))
    # plt.subplot(122)
    # plt.imshow(cv2.drawKeypoints(right_image, right_keypoints_list[0], None, (0, 255, 0)))
    # plt.show()

    left_points, right_points, _ = seagate_utils.multi_scale_match(left_keypoints_list, right_keypoints_list, left_descriptors_list, right_descriptors_list, left_image, right_image)

    print(len(left_points))

    if len(left_points) > 300:

        # seagate_utils.plot_match_points(left_image, right_image, left_points, right_points, False)

        np.savez(match_points_dir + left_image_id + '=' + right_image_id,
                 left_points=left_points,
                 right_points=right_points)
