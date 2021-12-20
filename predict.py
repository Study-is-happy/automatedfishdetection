import os
import cv2
import json
import shutil
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import config
import utils
import reset_predict

print_results = {}
for category in config.categories:

    print_results[category] = 0

cfg = get_cfg()
cfg.merge_from_file(
    'detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')

cfg.MODEL.RESNETS.NORM = 'GN'
cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
cfg.MODEL.ROI_BOX_HEAD.NORM = 'GN'
cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4
cfg.MODEL.ROI_BOX_HEAD.FC = 1
cfg.MODEL.FPN.NORM = 'GN'
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.PIXEL_MEAN = [0, 0, 0]

cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_categories

cfg.INPUT.MIN_SIZE_TEST = config.INPUT_MIN_SIZE_TRAIN[-1]

cfg.MODEL.WEIGHTS = config.MODEL_WEIGHTS_TEST

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.MODEL_ROI_HEADS_SCORE_THRESH_TEST
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.MODEL_ROI_HEADS_NMS_THRESH_TEST

predictor = DefaultPredictor(cfg)

with open(config.project_dir + 'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

with open(config.project_dir + 'easy_gt/instances.json') as easy_instances_file:
    easy_gt_annotation_generator = utils.easy_gt_annotation_generator(
        json.load(easy_instances_file))

easy_gt_index_generator = utils.easy_gt_index_generator()

cache_annotations = []

images_dir = config.project_dir + 'update/images/'

vis_instances = {}

annotation_id = 0

for image_file_name in os.listdir(images_dir):

    image_id = os.path.splitext(image_file_name)[0]

    image = cv2.imread(images_dir + image_file_name)

    height, width, _ = image.shape

    output = predictor(image)
    print(image_file_name)

    output_instances = output['instances']
    fields = output_instances.get_fields()

    annotations = []

    for category_id, score, bbox in zip(fields['pred_classes'], fields['scores'], fields['pred_boxes']):

        category_id = category_id.item()
        score = score.item()
        bbox = bbox.tolist()

        utils.abs_to_rel(bbox, width, height)

        annotation = {'image_id': image_id, 'width': width, 'height': height, 'category_id': category_id, 'score': score,
                      'bbox': bbox}
        annotations.append(annotation)

    print(len(annotations))
    if len(annotations) > 0:

        if image_id in update_instances:
            update_annotations = update_instances[image_id]['annotations']
        else:
            update_annotations = None

        for index, annotation in enumerate(annotations):

            if annotation['score'] > 0:
                for unchecked_annotation in annotations[index + 1:]:
                    if utils.get_bboxes_iou(annotation['bbox'], unchecked_annotation['bbox']) > 0.5:
                        unchecked_annotation['score'] = 0

                if update_annotations is not None:
                    for update_annotation in update_annotations:
                        if utils.get_bboxes_iou(annotation['bbox'], update_annotation['bbox']) > 0.3:
                            annotation['score'] = 0
                            break

        annotations = [
            annotation for annotation in annotations if annotation['score'] > 0]

        if len(annotations) > 0:

            cache_annotations.extend(annotations)

            shutil.copy(images_dir + image_file_name,
                        config.project_dir + 'predict/images/')

            vis_instance = {
                'width': width, 'height': height, 'annotations': []}
            vis_instances[image_id] = vis_instance

            if update_annotations is not None:
                utils.write_json_file(
                    update_annotations, config.project_dir + 'predict/exist_annotations/' + image_id + '.json')
                for update_annotation in update_annotations:
                    vis_instance['annotations'].append({'category_id': update_annotation['category_id'],
                                                        'bbox': update_annotation['bbox']})
            for annotation in annotations:
                vis_instance['annotations'].append({'category_id': annotation['category_id'], 'score': annotation['score'],
                                                    'bbox': annotation['bbox']})
                print_results[config.categories[annotation['category_id']]] += 1

            while len(cache_annotations) >= config.predict_per_file:

                current_annotations = cache_annotations[:config.predict_per_file]

                easy_gt_index = next(easy_gt_index_generator)
                easy_gt_annotation = next(easy_gt_annotation_generator)
                shutil.copy(config.project_dir + 'easy_gt/images/' + easy_gt_annotation['image_id'] + '.jpg',
                            config.project_dir + 'predict/images/')
                current_annotations.insert(
                    easy_gt_index, easy_gt_annotation)

                annotations_file_path = config.project_dir + 'predict/annotations/' + str(annotation_id) + '.json'

                utils.write_json_file(
                    current_annotations, annotations_file_path)

                shutil.copy(annotations_file_path,
                            config.project_dir + 'predict/current_annotations/')

                cache_annotations = cache_annotations[config.predict_per_file:]

                annotation_id += 1

utils.write_json_file(
    cache_annotations, config.project_dir + 'predict/annotations/cache.json')

utils.write_json_file(
    vis_instances, config.project_dir + 'predict/instances.json')

with open(config.project_dir + 'predict/annotation_ids.csv', 'a') as annotation_ids_file:

    for i in range(annotation_id):
        annotation_ids_file.write(str(i) + '\n')

print(print_results)
