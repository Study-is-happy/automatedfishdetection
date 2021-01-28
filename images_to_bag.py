import cv2
import cv_bridge
import rosbag
import sensor_msgs.msg
import os

calib_dir = '/data/automatedfishdetection/seagate/calib/RL_16_06/'

bridge = cv_bridge.CvBridge()

bag = rosbag.Bag(calib_dir + 'calib.bag', 'w')


# def write_images(images_dir, image_topic):

#     for image_file_name in sorted(os.listdir(images_dir)):

#         image = cv2.imread(images_dir + image_file_name)

#         print(image.shape)

#         image_message = bridge.cv2_to_imgmsg(image, encoding='rgb8')

#         bag.write(image_topic, image_message)


# write_images(calib_dir + 'port/', '/port/image_raw')
# write_images(calib_dir + 'stbd/', '/stbd/image_raw')

port_images_dir = calib_dir + 'port/'
stbd_images_dir = calib_dir + 'stbd/'

for port_image_file_name, stbd_image_file_name in zip(sorted(os.listdir(port_images_dir)), sorted(os.listdir(stbd_images_dir))):

    port_image = cv2.imread(port_images_dir + port_image_file_name)
    stbd_image = cv2.imread(stbd_images_dir + stbd_image_file_name)

    print(port_image_file_name)
    print(stbd_image_file_name)

    port_image_message = bridge.cv2_to_imgmsg(port_image, encoding='rgb8')
    stbd_image_message = bridge.cv2_to_imgmsg(stbd_image, encoding='rgb8')

    bag.write('/port/image_raw', port_image_message)
    bag.write('/stbd/image_raw', stbd_image_message)

bag.close()
