from flask import Flask, render_template, Response, request, session
from detect_video_ui import VideoCamera
from detect_img_ui import ImageCamera
from flask_session import Session
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class inputs(object):
    def __init__(self):
        self.weight='./yolov4'
        self.file='./pizza_radmaker1'
        self.type='others'

curr_input=inputs()

#SESSION_TYPE='redis'
#Session(app)
#session['weight_input']='./checkpoints/yolov4'
#sesion['video_input']='./data/video/pizza_radmaker1'


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/setting_inputs',methods=['POST'])
def setting_inputs():
    weight=request.form['weight']
    file=request.form['file']
    curr_input.weight=weight
    curr_input.file=file
    if 'video' in curr_input.file:
        curr_input.type='video'
    elif 'image' in curr_input.file:
        curr_input.type='image'
    else:
        print('Please check the directory of the file')

    #curr_input.type=request.form['type']
    print('weight_input',curr_input.weight)
    print('file_input',curr_input.file)
    print('file_type',curr_input.type)
    return ('',204)
    #return "hello"


@app.route('/video_feed')
def video_feed():
    #input=r
    #print('image_flask',model)
    if curr_input.type=='video':
        cam_ins=VideoCamera(curr_input.weight,curr_input.file)
    elif  curr_input.type=='image':
        cam_ins=ImageCamera(curr_input.weight,curr_input.file)
    else:
        print('please check file type')


    
    return Response(gen(cam_ins),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)