# project_dir = '/data/automatedfishdetection/seagate/all/'
project_dir = '/media/auv/Seagate Desktop Drive/automatedfishdetection/seagate/all/'

# calib_dir = '/data/automatedfishdetection/seagate/calib/SH_17_09/'

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
#               'skates/sharks', 'rockfish', 'flatfish', 'skates', 'unknown', 'background']

categories = ['sponges', 'rockfish', 'corals', 'invertebrates', 'roundfish', 'flatfish', 'skates', 'unknown', 'background']

colors = ['white', 'red', 'orange', 'yellow', 'pink', 'purple', 'blue', 'grey', 'grey']

# categories = ['珊瑚', 'unknown']
# colors = ['red', 'grey']

num_categories = len(categories) - 1

train_update = True

MODEL_WEIGHTS_TRAIN = project_dir + 'outputs/model_0215999.pth'

MODEL_ROI_HEADS_SCORE_THRESH_TEST = 0.5
MODEL_ROI_HEADS_NMS_THRESH_TEST = 0.5

MODEL_WEIGHTS_TEST = project_dir + 'outputs/model_0215999.pth'

predict_per_file = 9
