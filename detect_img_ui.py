import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.keras.models import load_model

import time

from datetime import datetime

#flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
        #flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    #'path to weights file')
        #flags.DEFINE_integer('size', 416, 'resize images to')
        #flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
        #flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
        #flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
        #flags.DEFINE_string('output', './detections/', 'path to output folder')
        #flags.DEFINE_float('iou', 0.45, 'iou threshold')
        #flags.DEFINE_float('score', 0.50, 'score threshold')
        #flags.DEFINE_boolean('count', False, 'count objects within images')
        #flags.DEFINE_boolean('dont_show', False, 'dont show image output')
        #flags.DEFINE_boolean('info', False, 'print info on detections')
        #flags.DEFINE_boolean('crop', False, 'crop detections from images')
        #flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
        #flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

class ImageCamera(object):
    def __init__(self, weight_input,image_input):

        self.framework='tf'
        #self.weights='./checkpoints/yolov4-416'
        self.weights=weight_input
        self.size=416
        self.tiny=False
        self.model='yolov4'
        #self.images='./data/images/pizza_radmaker.png'
        self.images=image_input
        self.output='./detections/'
        self.iou=0.45
        self.score=0.5
        self.count=True
        self.dont_show=False
        self.info=False
        self.crop=False
        self.ocr=False
        self.plate=False
        self.is_time_count=False    #make time series plot
        self.recent_output=None    #stores the earliar output

        #defining dictionary to hold total object counts
        #self.object_count={}
        self.time_count={}
        # read in all class names from config

        #modify config file
        if self.weights=='./checkpoints/yolov4-416':
            cfg.YOLO.CLASSES="./data/classes/coco.names"
        elif self.weights=='./checkpoints/yolov4-custom_tire_2000-416':
            cfg.YOLO.CLASSES="./data/classes/custom_tire.names"
        elif self.weights=='./checkpoints/yolov4-obj_cup_last-416':
            cfg.YOLO.CLASSES="./data/classes/custom_cup.names"

        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        '''
        if 'coco' in cfg.YOLO.CLASSES:   #for coco dataset, only considering pizza
            self.object_count['pizza']=0
            print('allowed', cfg.YOLO.CLASSES  )
            self.time_count['x']=[]
            self.time_count['y']=[]
        else:
            self.time_count['x']=[]
            self.time_count['y']=[]
            for ele in allowed_classes:
                self.object_count[ele] =0
        '''

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        #session = InteractiveSession(config=config)
        #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config_new(self.model, self.tiny)
        self.input_size = self.size
        images = self.images

        print('img', images)

        

        # load model
        if self.framework == 'tflite':
                interpreter = tf.lite.Interpreter(model_path=self.weights)
        else:
                saved_model_loaded = load_model(self.weights)
                self.infer = saved_model_loaded.signatures['serving_default']

        # loop through images in list and run Yolov4 model on each
        #for count, image_path in enumerate(images, 1):
        #print('image_path',image_path)
        self.image_path=self.images
        print('img1',self.image_path)
        self.original_image = cv2.imread(self.images)
        #print('img2',original_image)

        self.frame_num = 0


    def __del__(self):
        self.original_image.release()



    def get_frame(self):
        

        
        if self.frame_num==1:   #return the same outpur after 1st frame
            return self.recent_output

        else:

        
            original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(original_image, (self.input_size, self.input_size))
            image_data = image_data / 255.
            
            # get image name by using split method
            image_name = self.image_path.split('/')[-1]
            image_name = image_name.split('.')[0]
            
            #print('img2', image_name)
            

            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)

            if self.framework == 'tflite':
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], images_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if self.model == 'yolov3' and self.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                
                batch_data = tf.constant(images_data)
                pred_bbox = self.infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            # run non max suppression on detections
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=self.iou,
                score_threshold=self.score
            )

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = original_image.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
            
            # hold all detection data in one variable
            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to allow detections for only people)
            #allowed_classes = ['person']

            # if crop flag is enabled, crop each detection and save it as new image
            if self.crop:
                crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)

            # if ocr flag is enabled, perform general text extraction using Tesseract OCR on object detection bounding box
            if self.ocr:
                ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox)

            # if count flag is enabled, perform counting of objects
            if self.count:
                # count objects found
                counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
                #print('counted_classes in main', counted_classes)
                # loop through dict and print
                for key, value in counted_classes.items():
                    print("Number of {}s: {}".format(key, value))
                    #if key in ['pizza', 'tire', 'cup', 'defective_cup']: 
                    #continue
                        #self.object_count[key]=self.object_count[key]+value   
                        #self.time_count['x'].append(self.frame_num)
                        #self.time_count['y'].append(value)
                                      #(frame, pred_bbox, self.object_count, self.time_count, self.is_time_count, self.info, counted_classes, allowed_classes=allowed_classes, read_plate=self.plate)
                image = utils.draw_bbox(original_image, pred_bbox, self.time_count, self.is_time_count, self.info, counted_classes, allowed_classes=allowed_classes, read_plate = self.plate)
            else:
                image = utils.draw_bbox(original_image, pred_bbox, self.info, allowed_classes=allowed_classes, read_plate = FLAGS.plate)
            
            image = Image.fromarray(image.astype(np.uint8))
            if not self.dont_show:
                #pass
                #image.show()
                
                #pix = np.array(image)
                #cv2.imshow('',pix)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                #cv2.imwrite(self.output + str(image_name)  + '.png', image)
                #time.sleep(10)
                ret, jpeg = cv2.imencode('.jpg', image)

            self.frame_num +=1
            self.recent_output=jpeg.tobytes()
            return jpeg.tobytes()
            

#if __name__ == '__main__':
    #try:
        #app.run(main)
    #except SystemExit:
        #pass
