from flask import Flask, request
from utils import predict_only
from joblib import load

app = Flask(__name__)

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


@app.route("/compare_images/", methods=['POST'])
def predict_and_compare(model_path):

    image1 = request.files['image1']
    image2 = request.files['image2']
    
    img1 = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    img2 = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)

    model = load(model_path)

    prediction1 = model.predict(img1)

    prediction2 = model.predict(img2)

    if prediction1 == prediction2 :
        return "true"

    else:
        return "false"