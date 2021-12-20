import numpy as np
import cv2
import glob
import json
import shutil
import os
import matplotlib.pyplot as plt
import shapely.geometry

from MultiHarrisZernike import MultiHarrisZernike
from My_detector_3 import My_detector

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import config
import utils
import seagate_utils

seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/RL-16_06/d20161027_7/'

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')

left_dir = seagate_dir + 'port/port_rectified/'
right_dir = seagate_dir + 'stbd/stbd_rectified/'

left_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
right_mask = cv2.imread(config.calib_dir + 'stbd_mask.png', -1)

image_shape = np.flip(unrectify_params['image_size'])

right_unrectify_mask = cv2.remap(right_mask, unrectify_params['inv_right_mapx'], unrectify_params['inv_right_mapy'], cv2.INTER_NEAREST)
right_unrectify_contour, _ = cv2.findContours(right_unrectify_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
right_unrectify_polygon = shapely.geometry.Polygon(np.squeeze(right_unrectify_contour))

left_roi = seagate_utils.get_roi_mask(image_shape, unrectify_params['left_roi'])
right_roi = seagate_utils.get_roi_mask(image_shape, unrectify_params['right_roi'])


zernike = MultiHarrisZernike(Nfeats=41 * 48 * 20, seci=41, secj=48, levels=12, ratio=0.75,
                             sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)

# zernike = MultiHarrisZernike(Nfeats=10000, seci=5, secj=4, levels=6, ratio=0.75,
#                              sigi=2.75, sigd=1.0, nmax=8, lmax_nd=3, harris_threshold=None)
my_detector = My_detector()

cfg = get_cfg()
cfg.merge_from_file(
    '../detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')

cfg.MODEL.RESNETS.NORM = 'GN'
cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
cfg.MODEL.ROI_BOX_HEAD.NORM = 'GN'
cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4
cfg.MODEL.ROI_BOX_HEAD.FC = 1
cfg.MODEL.FPN.NORM = 'GN'
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.PIXEL_MEAN = [0, 0, 0]

# cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_categories
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.INPUT.MIN_SIZE_TEST = config.INPUT_MIN_SIZE_TRAIN[-1]

cfg.MODEL.WEIGHTS = config.MODEL_WEIGHTS_TEST

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.MODEL_ROI_HEADS_SCORE_THRESH_TEST
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.MODEL_ROI_HEADS_NMS_THRESH_TEST

predictor = DefaultPredictor(cfg)


def get_predict_annotations(image):

    output = predictor(image)

    output_instances = output['instances']
    fields = output_instances.get_fields()

    annotations = []

    for category_id, score, bbox in zip(fields['pred_classes'], fields['scores'], fields['pred_boxes']):

        category_id = category_id.item()
        score = score.item()
        bbox = bbox.tolist()

        annotation = {'category_id': category_id, 'score': score, 'bbox': bbox}

        annotations.append(annotation)

    return annotations


def get_bbox_polygon(bbox):
    return shapely.geometry.Polygon([bbox[0:2], [bbox[0], bbox[3]], bbox[2:4], [bbox[2], bbox[1]]])


def get_transform_point(homography_matrix, point):
    transform_point = homography_matrix.dot(np.append(point, [1]))
    return (transform_point / transform_point[-1])[:2]


with open(config.project_dir + 'all/instances_all.json') as instances_file:
    instances_dict = json.load(instances_file)

for left_image_path, right_image_path in zip(sorted(glob.glob(left_dir + '*.jpg')),
                                             sorted(glob.glob(right_dir + '*.jpg'))):

    if left_image_path != left_dir + '20161027.174658.00181_rect_color.jpg':
        continue

    print(left_image_path)

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

    seagate_utils.plot_match_points(gray_left_image, gray_right_image, np.array([]), np.array([]))

    left_keypoints_list, left_descriptors_list = my_detector.detectAndCompute(gray_left_image, left_roi)
    right_keypoints_list, right_descriptors_list = my_detector.detectAndCompute(gray_right_image, right_roi)

    left_points, right_points, _ = seagate_utils.multi_scale_match(left_keypoints_list, right_keypoints_list, left_descriptors_list, right_descriptors_list, gray_left_image, gray_right_image)

    # left_points, right_points = seagate_utils.get_match_points(zernike, gray_left_image, gray_right_image, left_roi, right_roi)

    val_left_points = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    val_right_points = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

    if len(val_left_points) < 8:
        continue

    homography_matrix, homography_mask = cv2.findHomography(val_left_points, val_right_points, cv2.FM_RANSAC, 3.0)

    homography_mask = (homography_mask.ravel() == 0)
    val_left_points = val_left_points[homography_mask]
    val_right_points = val_right_points[homography_mask]

    bad_fundamental = True

    if len(val_left_points) >= 8:

        fundamental_matrix, fundamental_mask = cv2.findFundamentalMat(val_left_points, val_right_points, cv2.FM_RANSAC, 0.2)

        if fundamental_mask is not None:

            fundamental_mask = (fundamental_mask.ravel() == 1)
            val_left_points = val_left_points[fundamental_mask]
            val_right_points = val_right_points[fundamental_mask]

            print(len(val_left_points))

            if len(val_left_points) > 150:

                # seagate_utils.plot_match_points(left_image, right_image, val_left_points, val_right_points)

                # shutil.copy(left_image_path, config.calib_dir + config.calib_sub_dir + 'port_images/')
                # shutil.copy(right_image_path, config.calib_dir + config.calib_sub_dir + 'stbd_images/')

                # np.savez(config.calib_dir + config.calib_sub_dir + 'match_points/' + left_image_id + '_' + right_image_id,
                #          left_points=left_points,
                #          right_points=right_points)

                bad_fundamental = True

    if bad_fundamental and np.count_nonzero(homography_mask) > 200:

        if left_image_id in instances_dict:

            instance = instances_dict[left_image_id]

            left_bbox_list = []
            right_transform_polygon_list = []

            vis_left_image = left_image.copy()
            vis_right_image = right_image.copy()

            for annotation in instance['annotations']:
                if annotation['category_id'] == 1:

                    bbox = annotation['bbox']
                    utils.rel_to_abs(bbox, instance['width'], instance['height'])

                    left_bbox_points = np.array([bbox[0:2], bbox[2:4]])

                    left_bbox_points = cv2.undistortPoints(left_bbox_points, unrectify_params['left_projection'][:, :3], np.zeros(5), R=np.linalg.inv(unrectify_params['left_rotation']), P=unrectify_params['left_camera_matrix']).flatten()

                    left_bbox_list.append(left_bbox_points)

                    right_transform_bbox_points = np.concatenate((get_transform_point(homography_matrix, left_bbox_points[0:2]),
                                                                  get_transform_point(homography_matrix, left_bbox_points[2:4])))

                    right_transform_polygon = get_bbox_polygon(right_transform_bbox_points)

                    if right_unrectify_polygon.intersects(right_transform_polygon):
                        right_transform_polygon = right_unrectify_polygon.intersection(right_transform_polygon)

                    right_transform_polygon_list.append(right_transform_polygon)

                    vis_left_bbox_points = utils.get_rint(left_bbox_points)
                    cv2.rectangle(vis_left_image, tuple(vis_left_bbox_points[0:2]), tuple(vis_left_bbox_points[2:4]), (0, 255, 0), 3)
                    # cv2.polylines(vis_right_image, [utils.get_rint(right_transform_polygon.exterior.coords)], True, (255, 0, 0), 3)

            if len(left_bbox_list) > 0:

                right_bbox_list = []
                right_polygon_list = []

                for right_annotation in get_predict_annotations(right_image):

                    if right_annotation['category_id'] == 0:

                        right_bbox = right_annotation['bbox']

                        right_bbox_list.append(right_bbox)

                        right_polygon_list.append(get_bbox_polygon(right_bbox))

                        vis_right_bbox = utils.get_rint(right_bbox)

                        cv2.rectangle(vis_right_image, tuple(vis_right_bbox[0:2]), tuple(vis_right_bbox[2:4]), (0, 255, 0), 3)

                if len(right_bbox_list) > 0:

                    plt.subplot(121)
                    plt.imshow(vis_left_image)
                    plt.subplot(122)
                    plt.imshow(vis_right_image)
                    plt.show()

                    iou_matrix = np.full((len(left_bbox_list), len(right_bbox_list)), -1.0)

                    for left_index, right_transform_polygon in enumerate(right_transform_polygon_list):
                        for right_index, right_polygon in enumerate(right_polygon_list):

                            if right_transform_polygon.intersects(right_polygon):
                                iou_matrix[left_index, right_index] = right_transform_polygon.intersection(right_polygon).area / right_transform_polygon.union(right_polygon).area

                    # print(iou_matrix)

                    left_annotation_mask = np.zeros(image_shape, np.uint8)
                    right_annotation_mask = np.zeros(image_shape, np.uint8)

                    vis_left_image = left_image.copy()
                    vis_right_image = right_image.copy()

                    exist_annotation = False

                    while np.max(iou_matrix) > 0.3:
                        best_index = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

                        left_index = best_index[0]
                        right_index = best_index[1]

                        left_bbox = utils.get_rint(left_bbox_list[left_index])
                        right_bbox = utils.get_rint(right_bbox_list[right_index])

                        cv2.rectangle(left_annotation_mask, tuple(left_bbox[0:2]), tuple(left_bbox[2:4]), 255, -1)
                        cv2.rectangle(right_annotation_mask, tuple(right_bbox[0:2]), tuple(right_bbox[2:4]), 255, -1)

                        iou_matrix[left_index, :] = -1
                        iou_matrix[:, right_index] = -1

                        cv2.rectangle(vis_left_image, tuple(left_bbox[0:2]), tuple(left_bbox[2:4]), (0, 255, 0), 3)
                        cv2.rectangle(vis_right_image, tuple(right_bbox[0:2]), tuple(right_bbox[2:4]), (0, 255, 0), 3)

                        exist_annotation = True

                    # print(iou_matrix)

                    if exist_annotation:

                        left_annotation_mask[left_roi == 0] = 0
                        right_annotation_mask[right_roi == 0] = 0

                        left_keypoints_list, left_descriptors_list = my_detector.detectAndCompute(gray_left_image, left_annotation_mask)
                        right_keypoints_list, right_descriptors_list = my_detector.detectAndCompute(gray_right_image, right_annotation_mask)

                        match_points = seagate_utils.multi_scale_match(left_keypoints_list, right_keypoints_list, left_descriptors_list, right_descriptors_list, gray_left_image, gray_right_image)

                        # match_points = seagate_utils.get_match_points(zernike, gray_left_image, gray_right_image, left_annotation_mask, right_annotation_mask)

                        if match_points is not None:

                            left_points, right_points, _ = match_points

                            seagate_utils.plot_match_points(left_image, right_image, left_points, right_points)

                            # shutil.copy(left_image_path, config.calib_dir + config.calib_sub_dir + 'port_images/')
                            # shutil.copy(right_image_path, config.calib_dir + config.calib_sub_dir + 'stbd_images/')

                            # np.savez(config.calib_dir + config.calib_sub_dir + 'match_points/' + left_image_id + '_' + right_image_id,
                            #          left_points=left_points,
                            #          right_points=right_points)
