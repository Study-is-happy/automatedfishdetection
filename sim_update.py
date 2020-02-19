import json

import config
import util

print_results = {'fish': 0, 'starfish': 0, 'sponge': 0, 'background': 0}
easy_annotation_indexes = [9]

with open(config.project_dir+'pmfs/instances.json') as gt_instances_file:
    gt_instances = json.load(gt_instances_file)

with open(config.project_dir+'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

with open(config.project_dir+'predict/annotation_ids.csv') as annotations_file_names:
    next(annotations_file_names)

    for annotations_file_name in annotations_file_names:
        annotations_file_name = annotations_file_name.rstrip('\n')

        with open(config.project_dir+'predict/annotations/'+annotations_file_name+'.json') as predict_annotations_file:
            predict_annotations = json.load(predict_annotations_file)

            for index, predict_annotation in enumerate(predict_annotations):
                if index not in easy_annotation_indexes:

                    image_id = predict_annotation['image_id']

                    predict_bbox = predict_annotation['bbox']

                    gt_instance = gt_instances[image_id]

                    if image_id not in update_instances:
                        update_instances[image_id] = {
                            'width': gt_instance['width'], 'height': gt_instance['height'], 'annotations': []}

                    update_bboxes = update_instances[image_id]['annotations']

                    best_iou = 0.2
                    best_gt_bbox = None

                    for gt_bbox in gt_instance['annotations']:
                        iou = util.get_bboxes_iou(
                            predict_bbox, gt_bbox['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_bbox = gt_bbox

                    if best_gt_bbox == None:
                        category_id = config.categories.index('background')
                        update_bboxes.append({'category_id': category_id,
                                              'bbox': predict_bbox, 'difficult': 1})
                        print_results[config.categories[category_id]] += 1
                    else:
                        for update_bbox in update_bboxes:
                            if update_bbox['bbox'] == best_gt_bbox['bbox']:
                                break
                        else:
                            update_bboxes.append(best_gt_bbox)
                            print_results[config.categories[best_gt_bbox['category_id']]] += 1

util.write_json_file(
    update_instances, config.project_dir+'update/instances.json')

print(print_results)
