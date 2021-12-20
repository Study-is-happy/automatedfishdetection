import numpy as np
import json
import torch
import os

import torch.multiprocessing
import torch.distributed

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm

from my_detectron2.my_rpn import my_RPN
from my_detectron2.my_dataset_mapper import DatasetMapper
from evaluator import Evaluator

import utils
import config


setup_logger()
torch.set_printoptions(precision=2, threshold=2000, sci_mode=False)


def get_dicts(datasets_dir):

    with open(datasets_dir + 'instances.json') as instances_file:

        instances_dict = json.load(instances_file)

    dicts = []

    for image_id, instance in instances_dict.items():

        instance['image_id'] = image_id

        instance['file_name'] = datasets_dir + 'images/' + image_id + '.jpg'

        for annotation in instance['annotations']:

            utils.rel_to_abs(annotation['bbox'],
                             instance['width'], instance['height'])
            annotation['bbox_mode'] = BoxMode.XYXY_ABS

        dicts.append(instance)

    return dicts


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, DatasetMapper(cfg, True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):

        return Evaluator(dataset_name)


def my_train():

    cfg = get_cfg()
    cfg.merge_from_file(
        'detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')

    cfg.MODEL.RESNETS.NORM = 'GN'
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.ROI_BOX_HEAD.NORM = 'GN'
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_BOX_HEAD.FC = 1
    cfg.MODEL.FPN.NORM = 'GN'
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
    # cfg.MODEL.RPN.POSITIVE_FRACTION = 0.3

    # cfg.INPUT.CROP.ENABLED = True
    # cfg.INPUT.CROP.SIZE = [0.8, 0.8]

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.STEPS = (180000, 216000)
    cfg.SOLVER.MAX_ITER = 240000
    cfg.SOLVER.CHECKPOINT_PERIOD = 6000

    cfg.CUSTOM_CLS_LOSS_FACTOR = 1.0

    cfg.OUTPUT_DIR = config.project_dir + 'outputs/'

    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD
    # cfg.TEST.EVAL_PERIOD = 1

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_categories

    cfg.INPUT.MIN_SIZE_TRAIN = config.INPUT_MIN_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TEST = config.INPUT_MIN_SIZE_TRAIN[-1]

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.MODEL_ROI_HEADS_SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.MODEL_ROI_HEADS_NMS_THRESH_TEST

    if config.train_update:

        cfg.DATASETS.TRAIN = ['gt/']
        cfg.DATASETS.TEST = ['init/']
        cfg.CUSTOM_IGNORE_PROB = 0.99
        cfg.MODEL.WEIGHTS = config.MODEL_WEIGHTS_TRAIN
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'my_RPN'

    else:
        cfg.DATASETS.TRAIN = ['update/']
        cfg.DATASETS.TEST = ['gt/']

    for datasets_dir in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST:
        DatasetCatalog.register(datasets_dir, lambda datasets_dir=datasets_dir: get_dicts(
            config.project_dir + datasets_dir))
        MetadataCatalog.get(datasets_dir).set(thing_classes=config.categories[:-1])

    # print(cfg)

    trainer = Trainer(cfg)
    if config.train_update:
        checkpoint = trainer.checkpointer._load_file(cfg.MODEL.WEIGHTS)
        trainer.checkpointer._load_model(checkpoint)
    trainer.train()


def my_distributed_worker(local_rank, main_func):

    torch.distributed.init_process_group(
        backend="NCCL", init_method=None, world_size=2, rank=local_rank
    )

    comm.synchronize()

    torch.cuda.set_device(local_rank)

    process_group = torch.distributed.new_group(list(range(0, 2)))
    comm._LOCAL_PROCESS_GROUP = process_group

    main_func()


# print('-------start-------')
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '29500'

if __name__ == '__main__':
    torch.multiprocessing.spawn(
        my_distributed_worker,
        nprocs=2,
        args=(my_train,),
        daemon=False,
    )
