from flask import Flask, request

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
