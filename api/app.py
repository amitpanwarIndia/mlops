import base64
from flask import Flask, request
from joblib import load
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

model_path = "./model_view/svm_gamma:0.001_C:1.joblib"
model = load(model_path)

@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/sum/<x>/<y>")
def get_sum(x,y):
    return str(int(x) + int(y))

@app.route("/model/", methods=['POST'])
def predict_model():
    js = request.get_json()
    x=js['x']
    y=js['y']

    return str(int(x) + int(y))

@app.route("/predict", methods=['POST'])
def predict_digit():
    img = request.json['image']
    image_data = base64.b64decode(img)
    img = Image.open(io.BytesIO(image_data))
    #img = img.resize((8, 8))
    img_array = np.array(img)
    img_flattened = img_array.flatten()

    prediction = model.predict([img_flattened])

    result = {        
        'prediction': int(prediction[0])
    }

    return result