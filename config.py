project_dir = '/data/automatedfishdetection/seagate/rockfish_train/'
# project_dir = '/home/zhiyongzhang/datasets/fish_detection/seagate/'

results_name = 'results_1.csv'

aws_access_key_id = 'AKIAIRD5JIXH2T5ERTGA'
aws_secret_access_key = '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'

email_address = 'isec.neufr@gmail.com'
email_password = 'HSingh1!'
email_name = 'Field Robotics Lab'

INPUT_MIN_SIZE_TRAIN = (448, 512, 576, 640, 704, 768, 832, 896)

# categories = ['fish', 'starfish', 'sponge', 'background']
# categories = ['rockfish_unid', 'greenstriped_rockfish', 'dover_sole', 'rex_sole', 'eelpout_unid',
#                      'poacher_unid', 'sandpaper_skate', 'slender_sole', 'flatfish_unid', 'roundfish_unid',
#                      'english_sole', 'sharpchin_rockfish', 'lingcod', 'yellowtail_rockfish', 'spotted_ratfish',
#                      'longnose_skate', 'shortspine_thornyhead', 'darkblotched_rockfish', 'arrowtooth_flounder', 'petrale_sole',
#                      'rosethorn_rockfish', 'pacific_ocean_perch', 'thornydead_unid', 'thornyhead_unid', 'fish_unid',
#                      'sablefish',
#                      'background']

# categories = ['corals', 'sponges', 'invertebrates', 'roundfish',
#               'skates/sharks', 'rockfish', 'flatfish', 'skates', 'other', 'background']
categories = ['rockfish', 'background']

# colors = ['purple', 'white', 'orange', 'blue',
#           'green', 'red', 'pink', 'yellow','grey', 'grey']
colors = ['red']

num_categories = len(categories)-1

train_update = False

MODEL_WEIGHTS_TRAIN = project_dir + 'outputs/model_0.pth'

MODEL_ROI_HEADS_SCORE_THRESH_TEST = 0.7
MODEL_ROI_HEADS_NMS_THRESH_TEST = 0.7

MODEL_WEIGHTS_TEST = project_dir + 'outputs/model_final.pth'

annotation_per_file = 10
gt_indexes = [9]
predict_per_file = annotation_per_file - len(gt_indexes)
