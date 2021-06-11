# yolov4-Object-Detection-and-Custom-UI
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

A wide range of custom functions & UI code for the deployment of real-time detection & analysis of objects using YOLOv4. Here the detection of objects on conveyor belt & defective items are presented.

## Skills Employed
* Modeling Techniques: CNN object detection, Darknet53, Yolov4, Transfer Learning, Computer Vision.
* Image & Video processing techniques: Opencv, GStreamer
* Tech Stack: Python. 
* Libraries: Tensorflow 2.3.0, Keras, Scikit-Learn, matplotlib.
* GUI techniques: Flask, HTML, JavaScript, RTSP.   

## A demo of the detection & analysis of objects on conveyor belt!
<p align="center"><img src="https://github.com/saha0073/Yolov4-Object-Detection-and-Custom-UI/blob/main/saved_detections/pizza_radmaker1.gif" style="width:80%"\></p>

## Currently Supported Custom Functions and Flags
* Counting Objects (total objects and per class)
* Print Info About Each Detection (class, confidence, bounding box coordinates)
* Time-series plot for the detailed analysis (Creates the time-series plot on the object count in each frame)


## Models
Here we have 3 models for 3 different use cases. 
* `./checkpoints/yolov4-416`: For pizza detection on conveyor belt. The original Darknet weights, trained on COCO dataset.
* `./checkpoints/yolov4-custom_tire_2000-416`: For tire detection on conveyor belt. Trained on 1500 tire images, collected from Google Open Images dataset. 
* `./checkpoints/yolov4-obj_cup_last-416`: For broken cups detection from good ones. Downloaded 300 broken cups & good cups from Google images, labeled using labelbox, and then trained for 2 classes.

The original Darknet model weights are available online. Due to large size of the other model weights those are not uploaeded here, if you want the models weights or training datasets please let me know. 


## Getting Started
### Folder structure
* `main_ui.py` is the Flask app that launches the UI application and calls necessary backend functions
* `./templates/index.html` is responsible for the UI 
* If the user provides an image as input `detect_img_ui.py` is called for object detection in the image, and if user provides a video as input `detect_video_ui.py` is called for object detection in the video.
* `./saved_detections/` consists of quite a few examples of yolov4 detections on pizza & tire on conveyor belt and broken cups.    

<!--### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
-->

### Run the UI application
```
### Install the necessary dependencies from `requirements.txt`
### Launch the Flask app
python main_ui.py 
```

### GUI Navigation
* The Flask app will be running from 5000 port, so please go to `localhost:5000` in your browser
* Import the weights and image/video: You can select `yolov4-Pizza` in the weights dropdown, choose `pizza_radmaker1.mp4` for filename and press `Load weights & file` button, it will import the yolov4 weights and the pizza video in the backend. 
* Press `start object detection button`, the object detection will start in GUI in a few seconds.
* It creates the time-series analysis plot for video input files, this feature can be switched on/off by setting `is_time_count` to `True/False` in main_ui.py.



## A screenshot of the UI that detects defective items!
<p align="center"><img src="https://github.com/saha0073/Yolov4-Object-Detection-and-Custom-UI/blob/main/saved_detections/ui.png"\></p>


This project has been inspired by [AIGuys](https://github.com/theAIGuysCode), so I would like to thank him. If you have any question please feel free to connect me in [Linkedin](https://www.linkedin.com/in/subhodip-saha-li/)
Happy Learning!




 


