# TODO

remote = False

###########################################################################

aws_access_key_id = 'AKIAIRD5JIXH2T5ERTGA'
aws_secret_access_key = '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'

if remote:
    project_dir = '/home/ubuntu/fish_detection/'

    INPUT_MIN_SIZE_TRAIN = (512, 640, 768, 896, 1024)
else:
    project_dir = '/home/zhiyongzhang/datasets/fish_detection/'

    INPUT_MIN_SIZE_TRAIN = (128,)

categories = ['fish', 'starfish', 'sponge', 'background']

num_categories = len(categories)-1

MODEL_RPN_PRE_NMS_TOPK_TRAIN = 2000
MODEL_RPN_POST_NMS_TOPK_TRAIN = 1000

SOLVER_STEPS = (15000, 18000)
SOLVER_MAX_ITER = 20000

train_update = False

MODEL_WEIGHTS_TRAIN = project_dir + 'outputs/model_0.pth'

MODEL_RPN_POSITIVE_FRACTION = 0.5
MODEL_RPN_BATCH_SIZE_PER_IMAGE = 512

MODEL_RPN_LOSS_WEIGHT = 1.0

MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE = 512

CUSTOM_IGNORE_PROB = 0.5
CUSTOM_CLS_LOSS_FACTOR = 1.0

MODEL_RPN_PRE_NMS_TOPK_TEST = 1000
MODEL_RPN_POST_NMS_TOPK_TEST = 1000

MODEL_ROI_HEADS_SCORE_THRESH_TEST = 0.7
MODEL_ROI_HEADS_NMS_THRESH_TEST = 0.3

TEST_DETECTIONS_PER_IMAGE = 200

MODEL_WEIGHTS_TEST = project_dir + 'outputs/model_0.pth'
