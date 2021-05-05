import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
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
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

#flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
#flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    #'path to weights file')
#flags.DEFINE_integer('size', 416, 'resize images to')
#flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
#flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
#flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
#flags.DEFINE_string('output', None, 'path to output video')
#flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
#flags.DEFINE_float('iou', 0.45, 'iou threshold')
#flags.DEFINE_float('score', 0.50, 'score threshold')
#flags.DEFINE_boolean('count', False, 'count objects within video')
#flags.DEFINE_boolean('dont_show', False, 'dont show video output')
#flags.DEFINE_boolean('info', False, 'print info on detections')
#flags.DEFINE_boolean('crop', False, 'crop detections from images')
#flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
#flags.DEFINE_boolean('tabulation', False, 'print tabulation on screen')

class VideoCamera(object):
    def __init__(self,weight_input,video_input):

        self.framework='tf'
        #self.weights='./checkpoints/yolov4-416'
        self.weights=weight_input
        self.size=416
        self.tiny=False
        self.model='yolov4'
        #self.video='./data/video/pizza_radmaker1.mp4'
        self.video=video_input
        self.output=None
        self.output_format='XVID'
        self.iou=0.45
        self.score=0.5
        self.count=True
        self.dont_show=False
        self.info=False
        self.crop=False
        self.ocr=False
        self.plate=False
        self.is_time_count=True    #make time series plot

        self.input_size = self.size
        self.video_path = self.video
        # get video name by using split method
        #video_name = self.video_path.split('/')[-1]
        #video_name = video_name.split('.')[0]
        # begin video capture
        try:
            self.vid = cv2.VideoCapture(int(self.video_path))
        except:
            self.vid = cv2.VideoCapture(self.video_path)

        #defining dictionary to hold total object counts
        self.object_count={}
        self.time_count={}
        #self.num_frame=0

        #modify config file
        if self.weights=='./checkpoints/yolov4-416':
            cfg.YOLO.CLASSES="./data/classes/coco.names"
        elif self.weights=='./checkpoints/yolov4-custom_tire_2000-416':
            cfg.YOLO.CLASSES="./data/classes/custom_tire.names"
        elif self.weights=='./checkpoints/yolov4-obj_cup_last-416':
            cfg.YOLO.CLASSES="./data/classes/custom_cup.names"

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        if 'coco' in cfg.YOLO.CLASSES:   #for coco dataset, only considering pizza
            self.object_count['pizza']=0
            self.time_count['x']=[]
            self.time_count['y']=[]
            print('allowed', cfg.YOLO.CLASSES  )
        else:
            self.time_count['x']=[]
            self.time_count['y']=[]
            for ele in allowed_classes:
                self.object_count[ele] =0

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        #tf.compat.v1.disable_eager_execution()
        
        #self.session = InteractiveSession(config=config)
        #self.session=tf.compat.v1.Session(config=config)

        #tf.compat.v1.disable_eager_execution()
        #self.graph = tf.get_default_graph()
        #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config_new(self.model, self.tiny)
        
        
        if self.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        else:
            #set_session(self.session)
            #tf.compat.v1.keras.backend.set_session(self.session)
            #with first_graph.as_default(), first_session.as_default():
            #saved_model_loaded = tf.saved_model.load(self.weights, tags=[tag_constants.SERVING])
            saved_model_loaded=load_model(self.weights)
            self.infer = saved_model_loaded.signatures['serving_default']
            #print('infer', self.infer)

        
        #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        

        
        out = None

        if self.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*self.output_format)
            out = cv2.VideoWriter(self.output, codec, fps, (width, height))

        self.frame_num = 0


    def __del__(self):
        self.vid.release()

    def get_frame(self):
        
        #while True:
        return_value, frame = self.vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            #break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if self.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if self.model == 'yolov3' and self.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([self.input_size, self.input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([self.input_size, self.input_size]))
        else:
            #tf.compat.v1.disable_eager_execution()
            #with self.session.as_default():
            #tf.compat.v1.keras.backend.set_session(self.session)
            #set_session(self.session)
                #tf.compat.v1.disable_eager_execution()
            #print('infer',self.infer)
            batch_data = tf.constant(image_data)
            pred_bbox = self.infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

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
        original_h, original_w, _ = frame.shape
        #print('boxes',boxes)
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

        # if crop flag is enabled, crop each detection and save it as new image
        if self.crop:
            crop_rate = 150 # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass          
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            else:
                pass

        if self.count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
                if key in ['pizza', 'tire', 'cup', 'defective_cup']: 
                #continue
                    self.object_count[key]=self.object_count[key]+value
                    self.time_count['x'].append(self.frame_num)
                    self.time_count['y'].append(value)

                #print("# of time {}s detected so far: {}".format(key, object_count[key]))

            image = utils.draw_bbox(frame, pred_bbox, self.object_count, self.time_count, self.is_time_count, self.info, counted_classes, allowed_classes=allowed_classes, read_plate=self.plate)
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        
        
        fps = 1.0 / (time.time() - start_time)

        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        #cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        

        if not self.dont_show:
            #cv2.imshow("result", result)
            ret, jpeg = cv2.imencode('.jpg', result)
            #self.num_frame=self.num_frame+1
            return jpeg.tobytes()
        
        #if self.output:
            #out.write(result)
        #if cv2.waitKey(1) & 0xFF == ord('q'): break
        #cv2.destroyAllWindows()

#if __name__ == '__main__':
    #try:
        #app.run(main)
    #except SystemExit:
        #pass
