from flask import Flask, request, jsonify, render_template
import os
from fastai.basic_train import load_learner
from fastai.vision import open_image
from flask_cors import CORS,cross_origin
app = Flask(__name__, static_url_path='/static')
CORS(app, support_credentials=True)

learn = load_learner(path='./models', file='poke_calsf_rsn50.pkl')
classes = learn.data.classes

def predict(img_file):
    prediction = learn.predict(open_image(img_file))
    probs_list = prediction[2].numpy()

    return {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    }

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

@app.route('/')
def render_page():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, port=os.getenv('PORT',5000))
