# TODO

remote = True

###########################################################################

aws_access_key_id = 'AKIAIRD5JIXH2T5ERTGA'
aws_secret_access_key = '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'

if remote:
    project_dir = '/home/ubuntu/fish_detection/'

    INPUT_MIN_SIZE_TRAIN = (512, 640, 768, 896)
else:
    project_dir = '/home/zhiyongzhang/datasets/fish_detection/'

    INPUT_MIN_SIZE_TRAIN = (128,)

categories = ['fish', 'starfish', 'sponge', 'background']

num_categories = len(categories)-1

train_update = True

MODEL_WEIGHTS_TRAIN = project_dir + 'outputs/model_0.pth'

MODEL_ROI_HEADS_SCORE_THRESH_TEST = 0.7
MODEL_ROI_HEADS_NMS_THRESH_TEST = 0.3

MODEL_WEIGHTS_TEST = project_dir + 'outputs/model_2.pth'
