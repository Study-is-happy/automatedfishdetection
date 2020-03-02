import numpy as np
import json
import config
import util

with open(config.project_dir+'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

with open(config.project_dir+'gt/instances.json') as gt_instances_file:
    gt_instances = json.load(gt_instances_file)

npos_list = []
tp_list = []

for _ in range(config.num_categories+1):
    npos_list.append(0)
    tp_list.append([])

for image_id in gt_instances:

    gt_annotations = []

    for _ in range(config.num_categories+1):
        gt_annotations.append([])

    for gt_annotation in gt_instances[image_id]['annotations']:

        category_id = gt_annotation['category_id']

        gt_annotation['detected'] = False

        gt_annotations[category_id].append(
            gt_annotation)

        npos_list[category_id] += 1

    if image_id in update_instances:

        for update_annotation in update_instances[image_id]['annotations']:

            category_id = update_annotation['category_id']

            best_iou = 0.5
            best_gt_annotation = None

            for gt_annotation in gt_annotations[category_id]:

                iou = util.get_bboxes_iou(
                    update_annotation['bbox'], gt_annotation['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_annotation = gt_annotation

                if best_gt_annotation is None or best_gt_annotation['detected']:
                    tp_list[category_id].append(0)
                else:
                    best_gt_annotation['detected'] = True
                    tp_list[category_id].append(1)

for index in range(config.num_categories):

    recall = np.sum(tp_list[index])/npos_list[index]

    print(recall)
