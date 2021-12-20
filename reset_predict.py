import os
import shutil

import config
import utils

predict_dir = config.project_dir + 'predict/'

shutil.rmtree(predict_dir)

os.mkdir(predict_dir)
os.mkdir(predict_dir + 'images/')
os.mkdir(predict_dir + 'annotations/')
os.mkdir(predict_dir + 'current_annotations/')
os.mkdir(predict_dir + 'exist_annotations/')
with open(predict_dir + 'annotation_ids.csv', 'w') as predict_annotations_file:
    predict_annotations_file.write('annotation_id\n')
utils.write_json_file({}, config.project_dir + 'predict/instances.json')
