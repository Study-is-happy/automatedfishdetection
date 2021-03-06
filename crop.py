import cv2
import json

import config
import utils

# TODO: Set the dir

crop_dir = config.project_dir + 'all/'

###########################################################################

col_num = 3
row_num = 3


def get_crop_bbox_point(bbox_point, start_point, end_point):

    if bbox_point > end_point:
        return end_point - start_point

    elif bbox_point < start_point:
        return 0

    return bbox_point - start_point


def get_bbox_size(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


with open(crop_dir + 'instances.json') as instances_file:
    instances = json.load(instances_file)

crop_instances = {}

for image_id, instance in instances.items():

    image = cv2.imread(crop_dir + 'images/' + image_id + '.jpg')

    image_width = instance['width']
    image_height = instance['height']

    crop_image_width = image_width / col_num
    crop_image_height = image_height / row_num

    col_list = []
    row_list = []

    for col_index in range(col_num + 1):

        col_list.append(utils.get_rint(crop_image_width * col_index))

    for row_index in range(row_num + 1):

        row_list.append(utils.get_rint(crop_image_height * row_index))

    for col_index in range(col_num):
        start_col = col_list[col_index]
        end_col = col_list[col_index + 1]
        for row_index in range(row_num):
            start_row = row_list[row_index]
            end_row = row_list[row_index + 1]

            crop_image_bbox = [start_col, start_row, end_col, end_row]
            crop_image_width = end_col - start_col
            crop_image_height = end_row - start_row

            annotations = []

            for annotation in instance['annotations']:

                bbox = annotation['bbox'].copy()

                utils.rel_to_abs(bbox, image_width, image_height)

                if utils.get_bboxes_iou(crop_image_bbox, bbox) > 0:
                    crop_bbox = bbox.copy()
                    crop_bbox[0] = get_crop_bbox_point(bbox[0], start_col, end_col)
                    crop_bbox[1] = get_crop_bbox_point(bbox[1], start_row, end_row)
                    crop_bbox[2] = get_crop_bbox_point(bbox[2], start_col, end_col)
                    crop_bbox[3] = get_crop_bbox_point(bbox[3], start_row, end_row)

                    if get_bbox_size(crop_bbox) / get_bbox_size(bbox) > 0.3:

                        utils.abs_to_rel(crop_bbox, crop_image_width, crop_image_height)

                        annotations.append({'category_id': annotation['category_id'], 'bbox': crop_bbox})

            if len(annotations) > 0:
                crop_image_id = image_id + '_' + str(col_index) + '_' + str(row_index)
                crop_instances[crop_image_id] = {'width': int(crop_image_width),
                                                 'height': int(crop_image_height),
                                                 'annotations': annotations}

                # cv2.imwrite(crop_dir + 'crop_images/' + crop_image_id + '.jpg', image[start_row:end_row, start_col:end_col])

utils.write_json_file(crop_instances, crop_dir + 'crop_instances.json')
