from xml.etree import ElementTree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# TODO: Set the dir

dataset_dir = '/home/zhiyongzhang/datasets/fish_detection/pacstorm_1/'

###########################################################################

annotations_dir = dataset_dir+'annotations/'

images_file_path = dataset_dir+'images.txt'

images_dir = dataset_dir+'images/'

colors = {'fish': 'red', 'starfish': 'orange',
          'sponge': 'white', 'background': 'grey'}

with open(images_file_path) as image_ids:

    for image_id in image_ids:

        image_id = image_id.rstrip('\n')

        image = mpimg.imread(images_dir+image_id+'.jpg')

        fig = plt.figure(figsize=(20, 20))

        plt.title(image_id)

        plt.imshow(image)

        current_axis = plt.gca()

        annotation_node = ElementTree.parse(
            annotations_dir+image_id+'.xml')

        object_nodes = annotation_node.findall('object')

        for object_node in object_nodes:

            category = object_node.find('name').text.lower()

            color = colors[category]

            bndbox_node = object_node.find('bndbox')

            bbox = [int(bndbox_node.find('xmin').text)-1,
                    int(bndbox_node.find('ymin').text)-1,
                    int(bndbox_node.find('xmax').text),
                    int(bndbox_node.find('ymax').text)]

            current_axis.add_patch(plt.Rectangle(
                (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], color=color, fill=False, linewidth=3))

            plt.text(bbox[0], bbox[1]-3,
                     category, color='white', size=30, bbox={'facecolor': color, 'alpha': 0.5, 'pad': 3})

        plt.show()
