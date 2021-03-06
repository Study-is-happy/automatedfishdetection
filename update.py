import csv
import json

import util
import config

results_approve_path = config.project_dir + \
    'results_approve/' + config.results_name

with open(config.project_dir + 'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

with open(results_approve_path) as results_file:
    results = csv.reader(results_file)
    next(results)
    for result in results:

        conf_indexes = json.loads(result[-4])

        result_annotations = json.loads(result[-8])

        for conf_index in conf_indexes:
            result_annotation = result_annotations[conf_index]
            image_id = result_annotation['image_id']
            if image_id not in update_instances:
                update_instances[image_id] = {
                    'width': result_annotation['width'], 'height': result_annotation['height'], 'annotations': []}

            update_annotations = update_instances[image_id]['annotations']

            update_annotations.append({
                'category_id': result_annotation['category_id'], 'bbox': result_annotation['bbox']})

util.write_json_file(
    update_instances, config.project_dir + 'update/instances.json')
