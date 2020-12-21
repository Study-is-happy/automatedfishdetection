import os

import config
import util

instances_file_path = config.project_dir + 'update/instances.json'

if os.path.exists(instances_file_path):
    os.remove(instances_file_path)
util.write_json_file({}, instances_file_path)
