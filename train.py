import numpy as np
import json
import torch
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

from my_rpn import my_RPN
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

cfg.DATASETS.TEST = ['test/']

cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'my_RPN'
cfg.MODEL.RESNETS.NORM = "GN"
cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"
cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4
cfg.MODEL.ROI_BOX_HEAD.FC = 1
cfg.MODEL.FPN.NORM = "GN"
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.PIXEL_MEAN = [0, 0, 0]

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.CHECKPOINT_PERIOD = 2
cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD

cfg.INPUT.MIN_SIZE_TEST = 1024

cfg.OUTPUT_DIR = config.project_dir+'outputs/'

cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_categories

cfg.INPUT.MIN_SIZE_TRAIN = config.INPUT_MIN_SIZE_TRAIN

cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = config.MODEL_RPN_PRE_NMS_TOPK_TRAIN
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = config.MODEL_RPN_POST_NMS_TOPK_TRAIN
cfg.MODEL.RPN.POSITIVE_FRACTION = config.MODEL_RPN_POSITIVE_FRACTION
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = config.MODEL_RPN_BATCH_SIZE_PER_IMAGE
cfg.MODEL.RPN.LOSS_WEIGHT = config.MODEL_RPN_LOSS_WEIGHT
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE

cfg.SOLVER.STEPS = config.SOLVER_STEPS
cfg.SOLVER.MAX_ITER = config.SOLVER_MAX_ITER

cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = config.MODEL_RPN_PRE_NMS_TOPK_TEST
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = config.MODEL_RPN_POST_NMS_TOPK_TEST

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.MODEL_ROI_HEADS_SCORE_THRESH_TEST
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.MODEL_ROI_HEADS_NMS_THRESH_TEST

cfg.TEST.DETECTIONS_PER_IMAGE = config.TEST_DETECTIONS_PER_IMAGE


cfg.CUSTOM_IGNORE_PROB = config.CUSTOM_IGNORE_PROB
cfg.CUSTOM_CLS_LOSS_FACTOR = config.CUSTOM_CLS_LOSS_FACTOR

if config.train_update:

    cfg.DATASETS.TRAIN = ['update/']
    cfg.MODEL.WEIGHTS = config.MODEL_WEIGHTS_TRAIN

else:
    cfg.DATASETS.TRAIN = ['train/']


for datasets_dir in cfg.DATASETS.TRAIN+cfg.DATASETS.TEST:
    DatasetCatalog.register(datasets_dir, lambda datasets_dir=datasets_dir: get_dicts(
        config.project_dir+datasets_dir))
    MetadataCatalog.get(datasets_dir).set(thing_classes=config.categories[:-1])


class Trainer(DefaultTrainer):

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