import numpy as np
import cv2
import csv
import json
import glob
import matplotlib.pyplot as plt
import os
import sklearn.cluster

from MultiHarrisZernike import MultiHarrisZernike

import config
import utils
import seagate_utils

seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-19-02/d20191009_1/'

left_dir = seagate_dir + 'port/port_rectified/'
right_dir = seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
stereo_params = np.load(config.calib_dir + config.calib_sub_dir + 'stereo_params.npz')

left_rectify_mask = cv2.remap(left_mask, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_NEAREST)
right_rectify_mask = cv2.remap(right_mask, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_NEAREST)

left_rectify_mask = cv2.remap(left_rectify_mask, stereo_params['left_mapx'], stereo_params['left_mapy'], cv2.INTER_NEAREST)
right_rectify_mask = cv2.remap(right_rectify_mask, stereo_params['right_mapx'], stereo_params['right_mapy'], cv2.INTER_NEAREST)

image_size = tuple(unrectify_params['image_size'])
image_shape = np.flip(image_size)

left_roi = seagate_utils.get_roi_mask(image_shape, stereo_params['left_roi'])
right_roi = seagate_utils.get_roi_mask(image_shape, stereo_params['right_roi'])

zernike = MultiHarrisZernike(Nfeats=10000, seci=5, secj=4, levels=6, ratio=0.75,
                             sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

disparity_unit = 16
block_size = 9
channel_num = 3
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disparity_unit * 2 + 1, disparity_unit * 2 + 1))

cluster = sklearn.cluster.KMeans(n_clusters=2)

# annotation_block_size = 3
# dilate_kernel = np.ones((annotation_block_size * 2 + 1, annotation_block_size * 2 + 1))

with open(config.project_dir + 'rockfish_size.csv', 'w') as mean_size_file:

    mean_size_writer = csv.writer(mean_size_file)

    with open(config.project_dir + 'train/instances.json') as instances_file:

        instances_dict = json.load(instances_file)

        for left_image_path, right_image_path in zip(sorted(glob.glob(left_dir + '*.tif')),
                                                     sorted(glob.glob(right_dir + '*.tif'))):

            image_id = os.path.splitext(os.path.basename(left_image_path))[0]

            if image_id in instances_dict:

                # if image_id == '20191009.154112.00088_rect_color':

                print(image_id)

                instance = instances_dict[image_id]

                exist_category_count = 0
                for annotation in instance['annotations']:
                    if annotation['category_id'] == 1:
                        exist_category_count += 1

                if exist_category_count < 1:
                    continue

                left_image = cv2.imread(left_image_path)
                right_image = cv2.imread(right_image_path)

                left_image[left_mask == 0] = 0
                right_image[right_mask == 0] = 0

                left_image = cv2.remap(left_image, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_LINEAR)
                right_image = cv2.remap(right_image, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_LINEAR)

                left_image = cv2.remap(left_image, stereo_params['left_mapx'], stereo_params['left_mapy'], cv2.INTER_LINEAR)
                right_image = cv2.remap(right_image, stereo_params['right_mapx'], stereo_params['right_mapy'], cv2.INTER_LINEAR)

                vis_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

                gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
                gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

                left_points, right_points = seagate_utils.get_match_points(zernike, gray_left_image, gray_right_image, left_roi, right_roi)

                match_mask = np.abs(left_points[:, 1] - right_points[:, 1]) < 2

                left_points = left_points[:, 0][match_mask]
                right_points = right_points[:, 0][match_mask]

                # seagate_utils.plot_match_points(left_image, right_image, left_points, right_points)

                points_diff = left_points - right_points

                min_points_diff = int(np.min(points_diff))
                max_points_diff = int(np.max(points_diff))

                min_disparity = min_points_diff - disparity_unit * 2
                num_disparities = max_points_diff + disparity_unit * 2 - min_disparity
                num_disparities += disparity_unit - num_disparities % disparity_unit

                stereo_disparity = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                                         numDisparities=num_disparities,
                                                         blockSize=block_size,
                                                         P1=8 * channel_num * block_size**2,
                                                         P2=32 * channel_num * block_size**2,
                                                         disp12MaxDiff=1,
                                                         preFilterCap=32,
                                                         uniquenessRatio=15,
                                                         # speckleWindowSize=100,
                                                         # speckleRange=32,
                                                         mode=cv2.STEREO_SGBM_MODE_HH)

                disparity_map = np.float32(stereo_disparity.compute(left_image, right_image)) / disparity_unit

                disparity_mask = np.full(image_shape, 255, np.uint8)
                disparity_mask[:, :min_disparity + num_disparities] = 0

                disparity_mask[left_rectify_mask == 0] = 0

                for point_diff in range(min_points_diff, max_points_diff + 1):

                    affine_matrix = np.float32([[1, 0, point_diff],
                                                [0, 1, 0]])
                    current_right_rectify_mask = cv2.warpAffine(right_rectify_mask, affine_matrix, image_size, flags=cv2.INTER_NEAREST)

                    disparity_mask[current_right_rectify_mask == 0] = 0

                disparity_mask = cv2.erode(disparity_mask, erode_kernel)

                disparity_map[disparity_mask == 0] = min_disparity - 1

                vis_disparity_image = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                vis_disparity_image = cv2.cvtColor(vis_disparity_image, cv2.COLOR_GRAY2RGB)

                for annotation_index, annotation in enumerate(instance['annotations']):

                    if annotation['category_id'] == 1:

                        bbox = annotation['bbox']
                        utils.rel_to_abs(bbox, instance['width'], instance['height'])

                        rint_bbox = utils.get_rint(bbox)

                        annotation_mask = np.zeros(image_shape, np.uint8)

                        cv2.rectangle(annotation_mask, tuple(rint_bbox[0:2]), tuple(rint_bbox[2:4]), 255, -1)

                        annotation_mask = cv2.remap(annotation_mask, unrectify_params['inv_left_mapx'], unrectify_params['inv_left_mapy'], cv2.INTER_NEAREST)
                        annotation_mask = cv2.remap(annotation_mask, stereo_params['left_mapx'], stereo_params['left_mapy'], cv2.INTER_NEAREST)

                        vis_contour, _ = cv2.findContours(annotation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(vis_left_image, vis_contour, -1, (0, 255, 0), 3)
                        cv2.drawContours(vis_disparity_image, vis_contour, -1, (0, 255, 0), 2)

                        disparity_mask_points = disparity_mask[annotation_mask == 255]

                        if len(disparity_mask_points) > 0 and np.all(disparity_mask_points == 255):

                            disparity_points = disparity_map[annotation_mask == 255]
                            disparity_points = disparity_points[disparity_points > min_disparity]

                            # plt.hist(disparity_points, bins='auto')
                            # plt.show()

                            cluster.fit(disparity_points.reshape(-1, 1))

                            # print(cluster.cluster_centers_)
                            # print(np.abs(cluster.cluster_centers_[0] - cluster.cluster_centers_[1]))

                            if np.abs(cluster.cluster_centers_[0] - cluster.cluster_centers_[1]) > 5:

                                # annotation_min_disparities = int(np.min(disparity_points)) - disparity_unit
                                # annotation_num_disparities = int(np.max(disparity_points)) - annotation_min_disparities + disparity_unit
                                # annotation_num_disparities += disparity_unit - annotation_num_disparities % disparity_unit

                                disparity_points = disparity_points[cluster.labels_ == np.argmax(cluster.cluster_centers_)]

                                min_disparity_point = np.min(disparity_points)
                                max_disparity_point = np.max(disparity_points)
                                mean_disparity_point = np.mean(disparity_points)

                                vis_disparity_image[(annotation_mask == 255)
                                                    & (disparity_map >= min_disparity_point)
                                                    & (disparity_map <= max_disparity_point)] = (0, 255, 0)

                                # left_annotation_mask = cv2.dilate(annotation_mask, dilate_kernel)
                                # right_annotation_mask = np.zeros(image_shape, np.uint8)

                                # for disparity_point in range(annotation_min_disparities, annotation_min_disparities + annotation_num_disparities + 1):

                                #     affine_matrix = np.float32([[1, 0, -disparity_point],
                                #                                 [0, 1, 0]])
                                #     current_right_annotation_mask = cv2.warpAffine(left_annotation_mask, affine_matrix, image_size, flags=cv2.INTER_NEAREST)

                                #     right_annotation_mask[current_right_annotation_mask == 255] = 255

                                # left_annotation_image = left_image.copy()
                                # left_annotation_image[left_annotation_mask == 0] = 0
                                # right_annotation_image = right_image.copy()
                                # right_annotation_image[right_annotation_mask == 0] = 0

                                # annotation_stereo_disparity = cv2.StereoSGBM_create(minDisparity=annotation_min_disparities,
                                #                                                     numDisparities=annotation_num_disparities,
                                #                                                     blockSize=annotation_block_size,
                                #                                                     P1=8 * channel_num * annotation_block_size**2,
                                #                                                     P2=32 * channel_num * annotation_block_size**2,
                                #                                                     disp12MaxDiff=1,
                                #                                                     preFilterCap=32,
                                #                                                     uniquenessRatio=15,
                                #                                                     # speckleWindowSize=100,
                                #                                                     # speckleRange=32,
                                #                                                     mode=cv2.STEREO_SGBM_MODE_HH)

                                # annotation_disparity_map = np.float32(annotation_stereo_disparity.compute(left_annotation_image, right_annotation_image)) / disparity_unit
                                # annotation_disparity_map[left_annotation_mask == 0] = annotation_min_disparities - 1

                                # # plt.hist(annotation_disparity_map.flatten(), bins='auto')
                                # # plt.show()

                                # plt.subplot(121)
                                # plt.imshow(left_annotation_image)
                                # plt.subplot(122)
                                # plt.imshow(annotation_disparity_map)
                                # plt.show()

                                bbox_points = np.array([bbox[0:2], [bbox[0], bbox[3]], bbox[2:4], [bbox[2], bbox[1]]])

                                bbox_points = cv2.undistortPoints(bbox_points,
                                                                  unrectify_params['left_projection'][:, :3],
                                                                  np.zeros(5),
                                                                  R=np.linalg.inv(unrectify_params['left_rotation']),
                                                                  P=unrectify_params['left_camera_matrix'])

                                bbox_points = cv2.undistortPoints(bbox_points,
                                                                  unrectify_params['left_camera_matrix'],
                                                                  np.zeros(5),
                                                                  R=stereo_params['left_rotation'],
                                                                  P=stereo_params['left_projection'])

                                bbox_points = utils.get_rint(np.squeeze(bbox_points))

                                annotation_disparity_map = disparity_map.copy()

                                for bbox_point in bbox_points:
                                    annotation_disparity_map[bbox_point[1], bbox_point[0]] = mean_disparity_point

                                boundary_point_image = cv2.reprojectImageTo3D(annotation_disparity_map, stereo_params['Q'])

                                mean_size = (cv2.norm(boundary_point_image[bbox_points[0][1], bbox_points[0][0]] - boundary_point_image[bbox_points[2][1], bbox_points[2][0]]) +
                                             cv2.norm(boundary_point_image[bbox_points[1][1], bbox_points[1][0]] - boundary_point_image[bbox_points[3][1], bbox_points[3][0]])) / 2

                                # print(mean_size)
                                mean_size_writer.writerow([image_id, str(annotation_index), str(mean_size)])

                # plt.subplot(121)
                # plt.imshow(vis_left_image)
                # plt.subplot(122)
                # plt.imshow(vis_disparity_image)
                # plt.show()

                # # cv2.imwrite(config.calib_dir + config.calib_sub_dir + 'vis_disparity_image.png', disparity_image)


# mean_size_list = []
# with open(config.project_dir + 'rockfish_size.csv') as mean_size_file:

#     mean_size_reader = csv.reader(mean_size_file)
#     for mean_size in mean_size_reader:
#         mean_size_list.append(float(mean_size[2]))
#         if float(mean_size[2]) > 500:
#             print(mean_size)
#         # print(float(mean_size[2]))

# plt.hist(np.array(mean_size_list), bins='auto')
# plt.show()
