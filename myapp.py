from flask import Flask,render_template,request
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
app=Flask(__name__)
@app.route('/')
def get():
    return render_template("form.html")

@app.route('/predict',methods=['POST'])
def get_salary():
    poly=pickle.load(open('model_pl.pkl','rb'))
    model=pickle.load(open('model.pkl','rb'))
    query=[[float(request.form['text2'])]]
    x_query=poly.transform(query)
    sal=model.predict(x_query)
    return 'Hii: '+request.form["text1"]+'your predicted salary after'+request.form["text2"]+'Experience is'+str(sal)
if __name__=='__main__':
    app.run(debug=True)