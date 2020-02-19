import config
import util
# TODO: Set the path

results_path_1 = config.predict_image_file_path

results_path_2 = config.update_image_file_path

###########################################################################

for image_name_1 in util.read_image_file(results_path_1)[1:]:

    for image_name_2 in util.read_image_file(results_path_2):

        if image_name_1 == image_name_2:
            break

    else:
        print(image_name_1)
