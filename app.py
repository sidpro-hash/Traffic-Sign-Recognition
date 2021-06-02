from flask import Flask, url_for, render_template, request, flash
# from flask.globals import request
from flask_cors import CORS, cross_origin
# import pandas as pd
import numpy as np
import tensorflow as tf
from werkzeug.utils import redirect, secure_filename
import os
import uuid
from time import gmtime, strftime
from flask_mail import Mail, Message
# import time


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = 'sjbcxzsdc15xz6czc'
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'botofsmitpanchal@gmail.com'
app.config['MAIL_PASSWORD'] = 'botofsmitpanchal123'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail_send = Mail(app)


model = tf.keras.models.load_model('./models/model_best_final.h5')
classes = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons'
          }
def get_output(image_path):
    test_image = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb', target_size=(32, 32),interpolation='bicubic')
    test_image = tf.keras.preprocessing.image.img_to_array(test_image, data_format="channels_last") / 255.
    test_image = np.expand_dims(test_image, axis=0)
    scores = model.predict(test_image)
    preds = np.argmax(scores, axis = 1)
    return classes[preds[0]]


@app.route("/", methods = ["GET", "POST"])
def home():
    if request.method == "POST":
        # print("Entered")
        in_image = request.files["in_image"]
        filename = str(uuid.uuid4()) + '_' + str(strftime("%Y_%m_%d-%H_%M_%S", gmtime()))
        in_image.save(os.path.join('uploads', secure_filename(f"{filename}.{in_image.filename.split('.')[-1]}")))
        if os.path.exists(os.path.join('uploads', f"{filename}.{in_image.filename.split('.')[-1]}")):
            output = get_output(os.path.join('uploads', f"{filename}.{in_image.filename.split('.')[-1]}"))
            os.remove(os.path.join('uploads', f"{filename}.{in_image.filename.split('.')[-1]}"))
            
            if output:
                flash(f'This is sign of {output.capitalize()}!', 'success')
                return render_template("index.html")
            else:
                return render_template("index.html")
        else:
            flash(f'Please upload image first!', 'danger')
            return render_template("index.html")
        # print(output)
    return render_template('index.html')


if __name__ =="__main__":
    app.run(debug=True)
