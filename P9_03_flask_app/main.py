
from flask import Flask, request, jsonify
from predict import make_prediction
import json

app = Flask('app')

@app.route('/test', methods=['GET'])
def test():
    return 'ping ok'

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    mask = make_prediction(img)
  
    
    return jsonify({'msg': 'success', 'mask': mask.tolist()})


