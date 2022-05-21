import pandas as pd
from flask import Flask,render_template,request
import joblib



app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    try:
        rm=int(request.form['rm'])
        age=int(request.form['age'])
        ym=int(request.form['ym'])
        children = int(request.form['children'])
        religious = int(request.form['religious'])
        edu = int(request.form['edu'])
        occupation = int(request.form['occupation'])
        occupation_husb = int(request.form['occupation_husb'])

        # x_predict=np.array([[rm,age,ym,children,religious,edu,occupation,occupation_husb]])
        x_predict=pd.DataFrame([[rm,age,ym,children,religious,edu,occupation,occupation_husb]],columns=['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'educ',
       'occupation', 'occupation_husb'])

        pipeline=joblib.load('clfpipeline.pkl')

        prediction=pipeline.predict(x_predict)
        print(prediction)
        if prediction==1:
            return render_template('result.html', prediction='She is having an affair')
        else:
            return render_template('result.html', prediction='She does not have an affair')


    except Exception as e:
        return e



if __name__=="__main__":
    app.run(debug=True)
