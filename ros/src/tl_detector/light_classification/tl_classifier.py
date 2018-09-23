#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import tensorflow as tf


class TLClassifier(object):
    def __init__(self):
	
        self.detection_threshold = 0.10
	
	self.traffic_light_box = []
	#self.traffic_light_scores = []
	
        #TODO load classifier
        #https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        #https://github.com/ActivateState/gococo
	#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'  #not compatible with TensorFlow 1.3.0 which is the version installed on Carla
        MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
		
	self.detection_graph = tf.Graph()
		
	with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()#ioutil.ReadFile(PATH_TO_CKPT)
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
				
	    self.sess = tf.Session(graph=self.detection_graph)
			
	    # Definite input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.detections_num = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_localization(self, image):
        """Detect objects in image according to COCO labels
	Args:
            image : image provided by camer vehicle
	Returns:
	    boxes, classes (labels) and scores of each detection
	"""
		
	image_array = np.asarray(image, dtype="uint8")
		
	# Expand dimensions 
        image_expanded = np.expand_dims(image_array, axis=0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num_detections) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.detections_num], feed_dict={self.image_tensor: image_expanded})
		
	boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)	
	
	return boxes, classes, scores
	    
    def get_traffic_light_box(self, boxes, classes, scores):
	    
	traffic_light_boxes = []
        traffic_light_scores = []

	max_score_idx = -1
	max_score = 0
		
        #rospy.logwarn('Number of bounding boxe[s] detected : %s\n', len(boxes))

	for idx, classe in enumerate(classes):
	    if classe == 10:   # 10 is the label of traffic light bounding boxes
                #rospy.logwarn('Box trafficlight : %s\n', idx)
                #rospy.logwarn('boxes[idx] : %s\n', boxes[idx])
	        traffic_light_boxes.append(boxes[idx])
		traffic_light_scores.append(scores[idx])
		if scores[idx] > max_score:
		    max_score = scores[idx]
		    max_score_idx = idx 
				
	number_traffic_light_boxes = len(traffic_light_boxes)
        #rospy.logwarn('number_traffic_light_boxes : %s\n', number_traffic_light_boxes)
        #rospy.logwarn('max_score_idx : %s\n', max_score_idx)
        #rospy.logwarn('max_score : %s\n', max_score)
				
	if number_traffic_light_boxes > 0 and max_score > 0.4:
	    rospy.logwarn('Number of bounding boxe[s] detected : {0}\n', len(traffic_light_boxes))
	    return traffic_light_boxes[max_score_idx] # assuming that there is only one traffic light at the time in the real application
	else:
	    rospy.logwarn('No traffic light bounding box detected !\n')
	    return None#[0, 0, 0, 0]
		
    def get_classification(self, image, box):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction		
	height = image.shape[0]
        width = image.shape[1]

        #rospy.logwarn('box[0] : %s\n', box[0])
        #rospy.logwarn('box[1] : %s\n', box[1])
        #rospy.logwarn('box[2] : %s\n', box[2])
        #rospy.logwarn('box[3] : %s\n', box[3])

        box_pixel = np.array([int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)])#.astype(int)
		
        #rospy.logwarn('box_pixel[0] : %s\n', box_pixel[0])
        #rospy.logwarn('box_pixel[1] : %s\n', box_pixel[1])
        #rospy.logwarn('box_pixel[2] : %s\n', box_pixel[2])
        #rospy.logwarn('box_pixel[3] : %s\n', box_pixel[3])

	cropped_image = image[box_pixel[0]:box_pixel[2], box_pixel[1]:box_pixel[3]]
		
	final_size = (20, 80)
        img = cv2.resize(np.array(cropped_image), final_size, interpolation=cv2.INTER_LINEAR)
		
	# Detection of traffic light color based on the following site :
	# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
		
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		
	# define range of green color in HSV
	lower_green = np.array([80, 0, 40])
        upper_green = np.array([160,100,80])
	# Threshold the HSV image to get only green colors
	mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

	# lower mask red (0-10)
        lower_red = np.array([0,0,0])
        upper_red = np.array([10,255,255])
	# Threshold the HSV image to get only red colors - lower mask
        low_mask_red = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask red (170-180)
        lower_red = np.array([160,0,0])
        upper_red = np.array([180,255,255])
	# Threshold the HSV image to get only red colors - upper mask
        up_mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
		
	# Threshold the HSV image to get only red colors
	mask_red = low_mask_red + up_mask_red
		
	# Define rate of both green and red color in the image
	rate_green = np.count_nonzero(mask_green) / (final_size[0]*final_size[1]) 
	rate_red = np.count_nonzero(mask_red) / (final_size[0]*final_size[1]) 
		
	if rate_green > self.detection_threshold and rate_green > rate_red:
	    return 2 #green
	elif rate_red > self.detection_threshold:
	    return 0 #red
	else:
	    return 1 #yellow
		
    def pipeline_classification(self, image):
	    
	# Use deep neural network to return detection bounding boxes
	boxes, classes, scores = self.get_localization(image)
		
	box = self.get_traffic_light_box(boxes, classes, scores)
									   
	if box == None:
	    return TrafficLight.UNKNOWN
			
	state_light = self.get_classification(image, box)
		
	if state_light == 0:
            return TrafficLight.RED
        elif state_light == 2:
            return TrafficLight.GREEN 
	else:
            return TrafficLight.YELLOW
