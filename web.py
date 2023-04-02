from flask import Flask,render_template,request
import pickle
import numpy as np


app =Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template ('home.html')

@app.route('/predict',methods=['POST'])
def predict_iris():
    SL=float(request.form.get("sepal_length"))
    SW=float(request.form.get("sepal_width"))
    PL=float(request.form.get("petal_length"))
    PW=float(request.form.get("petal_width"))
   
    results=model.predict(np.array([SL,SW,PL,PW]).reshape(1,4))
        
     
    return render_template('home.html',prediction_text="The measurments you entered is matching to", result=results)

if __name__=='__main__':
    app.run()  

