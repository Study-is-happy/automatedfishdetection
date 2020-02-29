import os
import json


def write_json_file(obj, json_file_path):

    with open(json_file_path, 'w') as json_file:

        json.dump(obj, json_file, indent='\t')


def get_bboxes_iou(bbox1, bbox2):

    width1 = bbox1[2]-bbox1[0]

    width2 = bbox2[2]-bbox2[0]

    overlap_width = width1+width2 - (max(bbox1[2], bbox2[2]) -
                                     min(bbox1[0], bbox2[0]))

    height1 = bbox1[3]-bbox1[1]

    height2 = bbox2[3]-bbox2[1]

    overlap_height = height1+height2 - (max(bbox1[3], bbox2[3]) -
                                        min(bbox1[1], bbox2[1]))

    if overlap_width > 0 and overlap_height > 0:

        area1 = width1*height1

        overlap_area = overlap_width*overlap_height

        area2 = width2*height2

        iou = float(overlap_area) / \
            (area1+area2-overlap_area)

        return iou

    return 0


def abs_to_rel(bbox, image_width, image_height):

    bbox[0] /= image_width
    bbox[1] /= image_height
    bbox[2] /= image_width
    bbox[3] /= image_height


def rel_to_abs(bbox, image_width, image_height):

    bbox[0] *= image_width
    bbox[1] *= image_height
    bbox[2] *= image_width
    bbox[3] *= image_height


def norm_bbox(bbox, image_width, image_height):

    for i in range(4):
        if bbox[i] < 0:
            bbox[i] = 0
        if i == 0 or i == 2:
            if bbox[i] > image_width:
                bbox[i] = image_width
        else:
            if bbox[i] > image_height:
                bbox[i] = image_height


def calc_timer(edge_timer, corner_timer):
    return edge_timer + corner_timer*1.5


def easy_annotation_generator(instances):
    while True:

        for image_id, instance in instances.items():
            width = instance['width']
            height = instance['height']
            for annotation in instance['annotations']:

                if annotation['difficult'] == 0:
                    yield {'image_id': image_id, 'width': width, 'height': height, 'category_id': annotation['category_id'], 'score': 1,
                           'bbox': annotation['bbox']}