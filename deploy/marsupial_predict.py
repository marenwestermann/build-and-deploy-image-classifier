from flask import Flask, request
from pathlib import Path
from fastai.learner import load_learner
from PIL import Image
import numpy as np


app = Flask(__name__) #create Flask app instance


@app.route('/', methods=['POST']) # tell Flask what URL should trigger the function
def make_predictions():
	
    # access data and read image
    file = request.files['image']
    img = Image.open(file.stream)
    
    # convert image to numpy array
    img_np = np.array(img)

    # make prediction
    animal = learn_inf.predict(img_np)[0]
    if animal == 'brushtail': 
        confidence = learn_inf.predict(img_np)[2][0].numpy()
    else: 
        confidence = learn_inf.predict(img_np)[2][1].numpy()
    return f"animal: {animal}; confidence: {confidence:.2f}"

if __name__ == '__main__':
    path = Path()
    learn_inf = load_learner(path/'export.pkl') # load model
    app.run(debug=True,host='0.0.0.0')
