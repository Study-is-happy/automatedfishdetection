import numpy as np
import cv2
import os
import csv
import glob
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import sklearn.cluster

import seagate_utils
import utils
import config

match_points_dir = config.calib_dir + config.calib_sub_dir + 'all_match_points/'

match_points_list = sorted(os.listdir(match_points_dir))

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']
image_size = tuple(unrectify_params['image_size'])
image_shape = np.flip(image_size)

stereo_params = np.load(config.calib_dir + config.calib_sub_dir + 'stereo_params.npz')
left_rotation = stereo_params['left_rotation']
right_rotation = stereo_params['right_rotation']
left_projection = stereo_params['left_projection']
right_projection = stereo_params['right_projection']

seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-16_06/d20161027_7/'
left_dir = seagate_dir + 'port/port_rectified/'
right_dir = seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

left_rectify_mask = cv2.remap(left_mask, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_NEAREST)
right_rectify_mask = cv2.remap(right_mask, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_NEAREST)

left_rectify_mask = cv2.remap(left_rectify_mask, stereo_params['left_mapx'], stereo_params['left_mapy'], cv2.INTER_NEAREST)
right_rectify_mask = cv2.remap(right_rectify_mask, stereo_params['right_mapx'], stereo_params['right_mapy'], cv2.INTER_NEAREST)

disparity_unit = 16
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disparity_unit * 2 + 1, disparity_unit * 2 + 1))


with open(config.project_dir + 'train/instances.json') as instances_file:

    instances_dict = json.load(instances_file)

# with open(config.project_dir + 'rockfish_size.csv', 'w') as mean_size_file:

# mean_size_writer = csv.writer(mean_size_file)

for match_points_file in match_points_list:

    left_image_id, right_image_id = os.path.splitext(match_points_file)[0].split('=')

    if left_image_id in instances_dict:

        if left_image_id != '20161027.175058.00271_rect_color':
            continue

        print(left_image_id)

        with open(left_dir + left_image_id + '.fct') as fct_file:
            latlng = np.double(fct_file.readline().split(',')[:2])

        left_image = cv2.imread(left_dir + left_image_id + '.jpg')
        right_image = cv2.imread(right_dir + right_image_id + '.jpg')

        left_image[left_mask == 0] = 0
        right_image[right_mask == 0] = 0

        left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
        right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

        left_image = cv2.remap(left_image, stereo_params['left_mapx'], stereo_params['left_mapy'], cv2.INTER_LINEAR)
        right_image = cv2.remap(right_image, stereo_params['right_mapx'], stereo_params['right_mapy'], cv2.INTER_LINEAR)

        match_points = np.load(match_points_dir + match_points_file)

        left_points = match_points['left_points']
        right_points = match_points['right_points']

        left_points = np.squeeze(cv2.undistortPoints(left_points,
                                                     left_camera_matrix,
                                                     np.zeros(5),
                                                     R=left_rotation,
                                                     P=left_projection))

        right_points = np.squeeze(cv2.undistortPoints(right_points,
                                                      right_camera_matrix,
                                                      np.zeros(5),
                                                      R=right_rotation,
                                                      P=right_projection))

        points_diff = left_points - right_points

        points_diff = points_diff[:, 0][np.abs(points_diff[:, 1]) < 1.0]

        min_disparity = utils.get_rint(np.min(points_diff))
        max_disparity = utils.get_rint(np.max(points_diff))

        print(min_disparity, max_disparity - min_disparity)

        if max_disparity - min_disparity > 150:
            continue

        num_disparities = max_disparity - min_disparity + disparity_unit * 5
        num_disparities += disparity_unit - num_disparities % disparity_unit

        stereo_disparity = cv2.StereoSGBM_create(minDisparity=min_disparity - disparity_unit,
                                                 numDisparities=num_disparities,
                                                 blockSize=5,
                                                 P1=8 * 3 * 5**2,
                                                 P2=32 * 3 * 5**2,
                                                 # disp12MaxDiff=0,
                                                 # preFilterCap=32,
                                                 # uniquenessRatio=15,
                                                 mode=cv2.STEREO_SGBM_MODE_HH
                                                 )

        disparity_map = np.float32(stereo_disparity.compute(left_image, right_image)) / disparity_unit

        disparity_mask = np.full(image_shape, 255, np.uint8)

        disparity_mask[:, :min_disparity + num_disparities] = 0

        disparity_mask[left_rectify_mask == 0] = 0

        for point_diff in range(min_disparity, max_disparity + 1):

            affine_matrix = np.float32([[1, 0, point_diff],
                                        [0, 1, 0]])

            current_right_rectify_mask = cv2.warpAffine(right_rectify_mask, affine_matrix, image_size, flags=cv2.INTER_NEAREST)

            disparity_mask[current_right_rectify_mask == 0] = 0

        disparity_mask = cv2.erode(disparity_mask, erode_kernel)

        disparity_map[disparity_mask == 0] = min_disparity - 1

        vis_disparity_image = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vis_disparity_image = cv2.cvtColor(vis_disparity_image, cv2.COLOR_GRAY2RGB)

        instance = instances_dict[left_image_id]

        test_counter = 0

        for annotation_index, annotation in enumerate(instance['annotations']):

            if annotation['category_id'] == 1:

                bbox = annotation['bbox']
                utils.rel_to_abs(bbox, instance['width'], instance['height'])

                rint_bbox = utils.get_rint(bbox)

                annotation_mask = np.zeros(image_shape, np.uint8)

                cv2.rectangle(annotation_mask, tuple(rint_bbox[0:2]), tuple(rint_bbox[2:4]), 255, -1)

                annotation_mask = cv2.remap(annotation_mask, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_NEAREST)
                annotation_mask = cv2.remap(annotation_mask, stereo_params['left_mapx'], stereo_params['left_mapy'], cv2.INTER_NEAREST)

                disparity_mask_points = disparity_mask[annotation_mask == 255]

                if len(disparity_mask_points) > 0 and np.all(disparity_mask_points == 255):

                    disparity_points = disparity_map[annotation_mask == 255]
                    disparity_points = disparity_points[disparity_points > min_disparity]

                    cluster = sklearn.cluster.KMeans(n_clusters=2)

                    cluster.fit(disparity_points.reshape(-1, 1))

                    max_cluster_index = np.argmax(cluster.cluster_centers_)
                    min_cluster_index = 1 - max_cluster_index

                    if cluster.cluster_centers_[max_cluster_index] - cluster.cluster_centers_[min_cluster_index] > 5:

                        test_counter += 1

                        disparity_points = disparity_points[cluster.labels_ == max_cluster_index]

                        min_disparity_point = np.min(disparity_points)
                        max_disparity_point = np.max(disparity_points)
                        mean_disparity_point = np.mean(disparity_points)

                        annotation_mask = (annotation_mask == 255) & (disparity_map >= min_disparity_point) & (disparity_map <= max_disparity_point)

                        vis_disparity_image[annotation_mask] = (0, 255, 0)

                        bbox_points = np.array([bbox[0:2], [bbox[0], bbox[3]], bbox[2:4], [bbox[2], bbox[1]]])

                        bbox_points = cv2.undistortPoints(bbox_points,
                                                          left_projection[:, :3],
                                                          np.zeros(5),
                                                          R=np.linalg.inv(unrectify_params['left_rotation']),
                                                          P=left_camera_matrix)

                        bbox_points = cv2.undistortPoints(bbox_points,
                                                          left_camera_matrix,
                                                          np.zeros(5),
                                                          R=left_rotation,
                                                          P=left_projection)

                        bbox_points = utils.get_rint(np.squeeze(bbox_points))

                        annotation_disparity_map = disparity_map.copy()

                        for bbox_point in bbox_points:
                            annotation_disparity_map[bbox_point[1], bbox_point[0]] = mean_disparity_point

                        boundary_point_image = cv2.reprojectImageTo3D(annotation_disparity_map, stereo_params['Q'])

                        mean_size = (cv2.norm(boundary_point_image[bbox_points[0][1], bbox_points[0][0]] - boundary_point_image[bbox_points[2][1], bbox_points[2][0]]) +
                                     cv2.norm(boundary_point_image[bbox_points[1][1], bbox_points[1][0]] - boundary_point_image[bbox_points[3][1], bbox_points[3][0]])) / 2

                        # mean_size_writer.writerow([left_image_id, annotation_index, mean_size, latlng,
                        #                            boundary_point_image[bbox_points[0][1], bbox_points[0][0]],
                        #                            boundary_point_image[bbox_points[1][1], bbox_points[1][0]],
                        #                            boundary_point_image[bbox_points[2][1], bbox_points[2][0]],
                        #                            boundary_point_image[bbox_points[3][1], bbox_points[3][0]]])

        # cv2.imwrite(config.calib_dir + config.calib_sub_dir + 'disparity_images/' + left_image_id + '.png', vis_disparity_image)

        print(test_counter)
