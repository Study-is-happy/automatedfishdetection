import json

import config
import util

with open(config.project_dir + 'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

for image_id, update_instance in update_instances.items():
    util.write_json_file(
        update_instance['annotations'], config.project_dir + 'predict/exist_annotations/' + image_id + '.json')
