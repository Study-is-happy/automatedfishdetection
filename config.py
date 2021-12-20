project_dir = '/media/auv/Seagate Desktop Drive/automatedfishdetection/seagate/all/'
# project_dir = '/media/auv/Seagate Desktop Drive/automatedfishdetection/pacstorm_pmfs_raw/'

calib_dir = '/media/auv/Seagate Desktop Drive/automatedfishdetection/seagate/calib/SH-18-12/'
calib_sub_dir = 'd20181014_2/'
seagate_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/SH-18-12/'

results_name = 'results_1.csv'

INPUT_MIN_SIZE_TRAIN = (448, 512, 576, 640, 704, 768, 832, 896)
# INPUT_MIN_SIZE_TRAIN = (448, 512, 576, 640, 704, 768)

# categories = ['fish', 'starfish', 'sponge', 'background']
# colors = ['red', 'orange', 'white', 'gray']

categories = ['sponges', 'rockfish', 'corals', 'invertebrates',
              'roundfish', 'flatfish', 'skates', 'unknown',
              'background', 'rockfish']

colors = ['white', 'red', 'orange', 'white',
          'gray', 'gray', 'gray', 'gray',
          'gray', 'pink']

# categories = ['rockfish', 'background']

# categories = ['rockfish_unid', 'greenstriped_rockfish', 'dover_sole', 'rex_sole', 'eelpout_unid',
#                      'poacher_unid', 'sandpaper_skate', 'slender_sole', 'flatfish_unid', 'roundfish_unid',
#                      'english_sole', 'sharpchin_rockfish', 'lingcod', 'yellowtail_rockfish', 'spotted_ratfish',
#                      'longnose_skate', 'shortspine_thornyhead', 'darkblotched_rockfish', 'arrowtooth_flounder', 'petrale_sole',
#                      'rosethorn_rockfish', 'pacific_ocean_perch', 'thornydead_unid', 'thornyhead_unid', 'fish_unid',
#                      'sablefish',
#                      'background']


num_categories = len(categories) - 1

train_update = True

MODEL_WEIGHTS_TRAIN = project_dir + 'outputs/model_0.pth'

MODEL_ROI_HEADS_SCORE_THRESH_TEST = 0.5
MODEL_ROI_HEADS_NMS_THRESH_TEST = 0.5

MODEL_WEIGHTS_TEST = project_dir + 'outputs/model_0.pth'

predict_per_file = 9
