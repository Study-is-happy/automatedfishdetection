import numpy as np
import json
import torch
import os
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

from my_rpn import my_RPN
from my_dataset_mapper import DatasetMapper
from evaluator import Evaluator

import util
import config


setup_logger()
torch.set_printoptions(precision=2, threshold=2000, sci_mode=False)


def get_dicts(datasets_dir):

    with open(datasets_dir+'instances.json') as instances_file:

        instances_dict = json.load(instances_file)

    dicts = []

    for image_id, instance in instances_dict.items():

        instance['image_id'] = image_id

        instance['file_name'] = datasets_dir+'images/'+image_id+'.jpg'

        for annotation in instance['annotations']:

            del annotation['difficult']

            util.rel_to_abs(annotation['bbox'],
                            instance['width'], instance['height'])
            annotation['bbox_mode'] = BoxMode.XYXY_ABS

        dicts.append(instance)

    return dicts


cfg = get_cfg()
cfg.merge_from_file(
    'detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')

cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'my_RPN'
cfg.MODEL.RESNETS.NORM = "GN"
cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"
cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4
cfg.MODEL.ROI_BOX_HEAD.FC = 1
cfg.MODEL.FPN.NORM = "GN"
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.7
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.5]
cfg.MODEL.RPN.NMS_THRESH = 0.5

cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.SIZE = [0.8, 0.8]

cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.BASE_LR = 0.005
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.STEPS = (30000, 36000)
cfg.SOLVER.MAX_ITER = 40000
cfg.SOLVER.CHECKPOINT_PERIOD = 4000
cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD

cfg.OUTPUT_DIR = config.project_dir+'outputs/'

cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_categories

cfg.INPUT.MIN_SIZE_TRAIN = config.INPUT_MIN_SIZE_TRAIN
cfg.INPUT.MIN_SIZE_TEST = config.INPUT_MIN_SIZE_TRAIN[-1]

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.MODEL_ROI_HEADS_SCORE_THRESH_TEST
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.MODEL_ROI_HEADS_NMS_THRESH_TEST

cfg.CUSTOM_IGNORE_PROB = config.CUSTOM_IGNORE_PROB
cfg.CUSTOM_CLS_LOSS_FACTOR = config.CUSTOM_CLS_LOSS_FACTOR

if config.train_update:

    cfg.DATASETS.TRAIN = ['update/']
    cfg.DATASETS.TEST = ['gt/']
    cfg.MODEL.WEIGHTS = config.MODEL_WEIGHTS_TRAIN

else:
    cfg.DATASETS.TRAIN = ['train/']
    cfg.DATASETS.TEST = ['test/']

for datasets_dir in cfg.DATASETS.TRAIN+cfg.DATASETS.TEST:
    DatasetCatalog.register(datasets_dir, lambda datasets_dir=datasets_dir: get_dicts(
        config.project_dir+datasets_dir))
    MetadataCatalog.get(datasets_dir).set(thing_classes=config.categories[:-1])

print(cfg)


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, DatasetMapper(cfg, True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):

        return Evaluator(dataset_name)


trainer = Trainer(cfg)
if config.train_update:
    checkpoint = trainer.checkpointer._load_file(cfg.MODEL.WEIGHTS)
    trainer.checkpointer._load_model(checkpoint)
trainer.train()

if config.remote:
    os.system('sudo poweroff')
