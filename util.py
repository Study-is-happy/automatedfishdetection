import os
import json
import numpy as np


def write_json_file(obj, json_file_path):

    with open(json_file_path, 'w') as json_file:

        json.dump(obj, json_file, indent='\t')


def get_bboxes_iou(bbox1, bbox2):

    width1 = bbox1[2] - bbox1[0]

    width2 = bbox2[2] - bbox2[0]

    overlap_width = width1 + width2 - (max(bbox1[2], bbox2[2])
                                       - min(bbox1[0], bbox2[0]))

    height1 = bbox1[3] - bbox1[1]

    height2 = bbox2[3] - bbox2[1]

    overlap_height = height1 + height2 - (max(bbox1[3], bbox2[3])
                                          - min(bbox1[1], bbox2[1]))

    if overlap_width > 0 and overlap_height > 0:

        area1 = width1 * height1

        overlap_area = overlap_width * overlap_height

        area2 = width2 * height2

        iou = float(overlap_area) / \
            (area1 + area2 - overlap_area)

        return iou

    return 0


def get_bbox_size(bbox, image_width, image_height):
    return (bbox[2] - bbox[0]) * image_width * (bbox[3] - bbox[1]) * image_height


def filter_overlap_instance(instance, iou):

    annotations = instance['annotations']

    width = instance['width']
    height = instance['height']

    for index, annotation in enumerate(annotations):

        for unchecked_annotation in annotations[index + 1:]:
            if get_bboxes_iou(annotation['bbox'], unchecked_annotation['bbox']) > iou:
                if get_bbox_size(annotation['bbox'], width, height) < get_bbox_size(unchecked_annotation['bbox'], width, height):
                    unchecked_annotation['overlap'] = True
                else:
                    annotation['overlap'] = True

    instance['annotations'] = [annotation for annotation in annotations if 'overlap' not in annotation]

    return [annotation for annotation in annotations if 'overlap' in annotation]


def abs_to_rel(bbox, image_width, image_height):

    bbox[0] /= image_width
    bbox[1] /= image_height
    bbox[2] /= image_width
    bbox[3] /= image_height


def rel_to_abs(bbox, image_width, image_height):

    bbox[0] = bbox[0] * image_width
    bbox[1] = bbox[1] * image_height
    bbox[2] = bbox[2] * image_width
    bbox[3] = bbox[3] * image_height


def pascal_voc_abs_to_rel(bbox, image_width, image_height):

    bbox[0] = (bbox[0] - 1) / image_width
    bbox[1] = (bbox[1] - 1) / image_height
    bbox[2] /= image_width
    bbox[3] /= image_height


def pascal_voc_rel_to_abs(bbox, image_width, image_height):

    bbox[0] = bbox[0] * image_width + 1
    bbox[1] = bbox[1] * image_height + 1
    bbox[2] = bbox[2] * image_width
    bbox[3] = bbox[3] * image_height


def norm_rel_bbox(bbox):

    for i in range(4):
        if bbox[i] < 0:
            bbox[i] = 0
        elif bbox[i] > 1:
            bbox[i] = 1


def get_rint(num):
    return int(np.rint(num))


def easy_gt_index_generator():
    while True:
        for index in range(5, 10):
            yield index


def easy_gt_annotation_generator(instances):
    while True:
        for image_id, instance in instances.items():
            for index, annotation in enumerate(instance['annotations']):
                if annotation['iou'] > 0.5:
                    yield {'image_id': image_id, 'width': instance['width'], 'height': instance['height'], 'category_id': annotation['category_id'], 'gt_annotation_index': index, 'score': 1,
                           'bbox': annotation['bbox']}
