import os

import config
import utils

instances_file_path = config.project_dir + 'update/instances.json'

if os.path.exists(instances_file_path):
    os.remove(instances_file_path)
utils.write_json_file({}, instances_file_path)
