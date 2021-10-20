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

seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-19-02/d20191009_1/'

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

left_dir = seagate_dir + 'port/port_rectified/'
right_dir = seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

zernike = MultiHarrisZernike(Nfeats=10000, seci=5, secj=4, levels=6, ratio=0.75,
                             sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

with open(config.project_dir + 'all/instances_all.json') as instances_file:
    instances_dict = json.load(instances_file)

for left_image_path, right_image_path in zip(sorted(glob.glob(left_dir + '*.tif')),
                                             sorted(glob.glob(right_dir + '*.tif'))):

    # if left_image_path != left_dir + '20191009.165433.01738_rect_color.tif':
    #     continue

    # if left_image_path != left_dir + '20191009.170145.01900_rect_color.tif':
    #     continue

    print(left_image_path)

    left_image_id = os.path.splitext(os.path.basename(left_image_path))[0]
    right_image_id = os.path.splitext(os.path.basename(right_image_path))[0]

    if left_image_id in instances_dict:

        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)

        left_image[left_mask == 0] = 0
        right_image[right_mask == 0] = 0

        left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
        right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

        match_points = np.load(config.calib_dir + config.calib_sub_dir + 'match_points/' + left_image_id + '_' + right_image_id + '.npz')

        left_points = match_points['left_points']
        right_points = match_points['right_points']

        print(len(left_points))

        # seagate_utils.plot_match_points(left_image, right_image, left_points, right_points)

        points_mask = np.load(config.calib_dir + config.calib_sub_dir + 'match_points/mask_' + left_image_id + '_' + right_image_id + '.npz')

        homography_mask = points_mask['homography_mask']
        fundamental_mask = points_mask['fundamental_mask']

        left_points = left_points[homography_mask][fundamental_mask]
        right_points = right_points[homography_mask][fundamental_mask]

        print(len(left_points))

        seagate_utils.plot_match_points(left_image, right_image, left_points, right_points)
