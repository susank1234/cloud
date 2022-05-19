from flask import Flask, request, redirect, url_for, render_template
from keras.models import load_model 
import numpy as np 
from keras.preprocessing import image
import cv2
application = Flask(__name__)
global model 
model = load_model('model.h5') 
@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict_vechile():
    img_file=request.files.get('file')
    npimg = np.fromfile(img_file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    probabilities = model.predict(img)
 
    
    print(probabilities)
    number_to_class = ['aphids','armyworm','beetle','bollworm', 'grasshopper','mites',  'mosquito','sawfly',  'stem_borer']
    index = np.argmax(probabilities,axis=1)
    predictions = {
        "class":number_to_class[index[0]],
      }
    return render_template('predict.html', predictions=predictions)

if __name__=='__main__':
  application.run(debug=True)