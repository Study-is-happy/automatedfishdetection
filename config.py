# TODO

remote = False

###########################################################################

aws_access_key_id = 'AKIAIRD5JIXH2T5ERTGA'
aws_secret_access_key = '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'

if remote:
    project_dir = '/home/ubuntu/fish_detection/'

    INPUT_MIN_SIZE_TRAIN = (448, 512, 576, 640, 704, 768, 832, 896)
else:
    project_dir = '/data/fish_detection/'

    INPUT_MIN_SIZE_TRAIN = (128,)

categories = ['fish', 'starfish', 'sponge', 'background']
lynker_categories = ['rockfish_unid', 'greenstriped_rockfish', 'dover_sole', 'rex_sole', 'eelpout_unid',
                     'poacher_unid', 'sandpaper_skate', 'slender_sole', 'flatfish_unid', 'roundfish_unid',
                     'english_sole', 'sharpchin_rockfish', 'lingcod', 'yellowtail_rockfish', 'spotted_ratfish',
                     'longnose_skate', 'shortspine_thornyhead', 'darkblotched_rockfish', 'arrowtooth_flounder', 'petrale_sole',
                     'rosethorn_rockfish', 'pacific_ocean_perch', 'thornydead_unid', 'thornyhead_unid', 'fish_unid',
                     'sablefish',
                     'background']
                     
seagate_categories = ['Corals', 'Sponges', 'Invertebrates', 'Roundfish', 'Skates/Sharks', 'Rockfish', 'Flatfish', 'Other', 'Skates']                 

num_categories = len(categories)-1

train_update = False

MODEL_WEIGHTS_TRAIN = project_dir + 'outputs/model_0.pth'

MODEL_ROI_HEADS_SCORE_THRESH_TEST = 0.7
MODEL_ROI_HEADS_NMS_THRESH_TEST = 0.7

MODEL_WEIGHTS_TEST = project_dir + 'outputs/model_final.pth'
