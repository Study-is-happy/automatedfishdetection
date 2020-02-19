import os
import json
import shutil

import config
import util

if os.path.exists(config.project_dir):
    shutil.rmtree(config.project_dir)

os.mkdir(config.project_dir)
os.mkdir(config.project_dir+'train/')
os.mkdir(config.project_dir+'train/images/')
util.write_json_file({}, config.project_dir+'train/instances.json')
os.mkdir(config.project_dir+'test/')
os.mkdir(config.project_dir+'test/images/')
util.write_json_file({}, config.project_dir+'test/instances.json')
os.mkdir(config.project_dir+'outputs/')
os.mkdir(config.project_dir+'results/')
os.mkdir(config.project_dir+'predict/')
os.mkdir(config.project_dir+'predict/images/')
os.mkdir(config.project_dir+'predict/annotations/')
os.mkdir(config.project_dir+'predict/exist_annotations/')
with open(config.project_dir+'predict/annotation_ids.csv', 'w') as predict_annotations_file:
    predict_annotations_file.write('annotation_id\n')
util.write_json_file({}, config.project_dir+'predict/instances.json')
os.mkdir(config.project_dir+'update/')
os.mkdir(config.project_dir+'update/images/')
util.write_json_file({}, config.project_dir+'update/instances.json')
