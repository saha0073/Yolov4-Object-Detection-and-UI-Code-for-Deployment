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

## A screenshot of the UI that detects defective items!
<p align="center"><img src="https://github.com/saha0073/Yolov4-Object-Detection-and-Custom-UI/blob/main/saved_detections/ui.png"\></p>

## Currently Supported Custom Functions and Flags
* Counting Objects (total objects and per class)
* Print Info About Each Detection (class, confidence, bounding box coordinates)
* Time-series plot for the detailed analysis (Creates the time-series plot on the object count in each frame)

## Getting Started
### Conda (Recommended)

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

## YOLOv4 Using Tensorflow (tf, .pb model)
```
# Run yolov4 on video
python main_ui.py 

# The Flask app will be running from 5000 port, so please go to `localhost:5000` in your browser
# Import the weights and image/video: So `./checkpoints/yolov4-416` in place of weights, `./data/images/pizza_radmaker2.png` in file and press Load weights & file button, it will import the yolov4 weights and the pizza video to the backend. 
# Press start object detection, the object detection will start in GUI in few seconds.
```



 


