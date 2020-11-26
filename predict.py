import os
import cv2
import json
import shutil
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import config
import util
import reset_predict

annotation_per_file = 10
gt_indexes = [9]
predict_per_file = annotation_per_file-len(gt_indexes)

print_results = {'fish': 0, 'starfish': 0, 'sponge': 0}

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

cfg.INPUT.MIN_SIZE_TEST = 896

cfg.MODEL.WEIGHTS = config.MODEL_WEIGHTS_TEST

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.MODEL_ROI_HEADS_SCORE_THRESH_TEST
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.MODEL_ROI_HEADS_NMS_THRESH_TEST

predictor = DefaultPredictor(cfg)

with open(config.project_dir+'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

with open(config.project_dir+'easy/instances.json') as easy_instances_file:
    easy_annotation_generator = util.easy_annotation_generator(
        json.load(easy_instances_file))

cache_annotations = []

images_dir = config.project_dir+'update/images/'

vis_instances = {}

annotation_id = 0

for image_file_name in os.listdir(images_dir):

    image_id = os.path.splitext(image_file_name)[0]

    image = cv2.imread(images_dir+image_file_name)

    height, width, _ = image.shape

    output = predictor(image)

    output_instances = output['instances']
    fields = output_instances.get_fields()

    annotations = []

    for category_id, score, bbox in zip(fields['pred_classes'], fields['scores'], fields['pred_boxes']):

        category_id = category_id.item()
        score = score.item()
        bbox = bbox.tolist()

        util.abs_to_rel(bbox, width, height)

        annotation = {'image_id': image_id, 'width': width, 'height': height, 'category_id': category_id, 'score': score,
                      'bbox': bbox}
        annotations.append(annotation)

    if len(annotations) > 0:

        if image_id in update_instances:
            update_annotations = update_instances[image_id]['annotations']
        else:
            update_annotations = None

        for index, annotation in enumerate(annotations):

            if annotation['score'] > 0:
                for unchecked_annotation in annotations[index+1:]:
                    if util.get_bboxes_iou(annotation['bbox'], unchecked_annotation['bbox']) > 0.1:
                        unchecked_annotation['score'] = 0

                if update_annotations is not None:
                    for update_annotation in update_annotations:
                        if util.get_bboxes_iou(annotation['bbox'], update_annotation['bbox']) > 0.3:
                            annotation['score'] = 0
                            break

        annotations = [
            annotation for annotation in annotations if annotation['score'] > 0]

        if len(annotations) > 0:

            cache_annotations.extend(annotations)

            shutil.copy(images_dir+image_file_name,
                        config.project_dir+'predict/images/')

            vis_instance = {
                'width': width, 'height': height, 'annotations': []}
            vis_instances[image_id] = vis_instance

            if update_annotations is not None:
                util.write_json_file(
                    update_annotations, config.project_dir+'predict/exist_annotations/'+image_id+'.json')
                for update_annotation in update_annotations:
                    vis_instance['annotations'].append({'category_id': update_annotation['category_id'],
                                                        'bbox': update_annotation['bbox']})
            for annotation in annotations:
                vis_instance['annotations'].append({'category_id': annotation['category_id'], 'score': annotation['score'],
                                                    'bbox': annotation['bbox']})
                print_results[config.categories[annotation['category_id']]] += 1

            while len(cache_annotations) >= predict_per_file:

                current_annotations = cache_annotations[:predict_per_file]

                for gt_index in gt_indexes:
                    easy_annotation = next(easy_annotation_generator)
                    shutil.copy(config.project_dir+'easy/images/'+easy_annotation['image_id']+'.jpg',
                                config.project_dir+'predict/images/')
                    current_annotations.insert(
                        gt_index, easy_annotation)

                util.write_json_file(
                    current_annotations, config.project_dir+'predict/annotations/'+str(annotation_id)+'.json')

                cache_annotations = cache_annotations[predict_per_file:]

                annotation_id += 1

util.write_json_file(
    cache_annotations, config.project_dir+'predict/annotations/cache.json')

util.write_json_file(
    vis_instances, config.project_dir+'predict/instances.json')

with open(config.project_dir+'predict/annotation_ids.csv', 'a') as annotation_ids_file:

    for i in range(annotation_id):
        annotation_ids_file.write(str(i) + '\n')

print(print_results)
