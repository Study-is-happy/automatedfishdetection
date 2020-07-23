import numpy as np
import json
import config
import util

with open(config.project_dir+'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)
    # gt_instances = json.load(update_instances_file)

with open(config.project_dir+'fake_gt/instances.json') as gt_instances_file:
    gt_instances = json.load(gt_instances_file)
    # update_instances = json.load(gt_instances_file)

n_pos_list = []
n_true_pos_list = []

for _ in range(config.num_categories+1):
    n_pos_list.append(0)
    n_true_pos_list.append(0)

for image_id in gt_instances:

    gt_annotations = gt_instances[image_id]['annotations']

    for gt_annotation in gt_annotations:

        gt_annotation['detected'] = False

        n_pos_list[gt_annotation['category_id']] += 1

    if image_id in update_instances:

        update_annotations = update_instances[image_id]['annotations']

        for update_annotation in update_annotations:

            category_id = update_annotation['category_id']

            best_iou = 0.5
            best_gt_annotation = None

            for gt_annotation in gt_annotations:

                if category_id == gt_annotation['category_id']:

                    iou = util.get_bboxes_iou(
                        update_annotation['bbox'], gt_annotation['bbox'])

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_annotation = gt_annotation

                    if best_gt_annotation is not None and not best_gt_annotation['detected']:

                        best_gt_annotation['detected'] = True
                        n_true_pos_list[category_id] += 1

print(n_true_pos_list)

for index in range(config.num_categories):

    recall = n_true_pos_list[index]/n_pos_list[index]

    print(recall)

util.write_json_file(gt_instances, config.project_dir +
                     'evaluate_gt/instances.json')
