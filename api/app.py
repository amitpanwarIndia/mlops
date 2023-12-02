import base64
from flask import Flask, request
from joblib import load
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
#app.run(debug=True) Removed debugging

# print('here')
svm_model = None
tree_model = None
logistic_model = None

def load_model():
    global svm_model
    global tree_model
    global logistic_model

    svm_model_path = "./model_view/svm_gamma:0.001_C:1.joblib"
    svm_model = load(svm_model_path)

    tree_model_path = "./model_view/tree_max_depth:100.joblib"
    tree_model = load(tree_model_path)

    logistic_model_path = "./model_view/logistic_solver:liblinear.joblib"
    logistic_model = load(logistic_model_path)

load_model()

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

@app.route("/predict/<model_type>", methods=['POST'])
def predict_digit(model_type):

    img = request.json['image']
    image_data = base64.b64decode(img)
    img = Image.open(io.BytesIO(image_data))
    #img = img.resize((8, 8))
    img_array = np.array(img)
    img_flattened = img_array.flatten()

    if model_type == "svm":
        prediction = svm_model.predict([img_flattened])

    elif model_type == "tree":
        prediction = tree_model.predict([img_flattened])

    elif model_type == "lr":
        prediction = logistic_model.predict([img_flattened])

    result = {        
        'prediction': int(prediction[0])
    }

    return result
