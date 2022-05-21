import numpy as np
import pandas as pd
from src import CONFIG
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline,make_pipeline
import joblib


def train(data):
    df=pd.read_csv(data)
    df['affairs']=df['affairs'].map(lambda x:1 if x>0 else 0)
    x=df.drop('affairs',axis=1)
    y=df['affairs']
    print(x.columns)
    transformer=make_column_transformer(
       (OneHotEncoder(),["occupation", "occupation_husb"]),
        remainder='passthrough')

    clf = LogisticRegression(max_iter=2000)
    pipeline=Pipeline([
        ('transformer',transformer),
        ('classifier',clf)
    ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print(x_train.shape)
    print(y_train.shape)

    pipeline.fit(x_train,y_train)
    print(x_train.columns)
    # clf.fit(x_train,y_train)
    print('train score',pipeline.score(x_train,y_train))
    print('test score',pipeline.score(x_test,y_test))
    print('prediction',pipeline.predict(x_test))

    joblib.dump(pipeline,CONFIG.PIPELINE)



train(CONFIG.DATA)

#
def predict(data):
    df=pd.DataFrame(data,columns=['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'educ',
       'occupation', 'occupation_husb'])
    print('yes')
    pipeline=joblib.load(CONFIG.PIPELINE)
    pred=pipeline.predict(df)
    # x_predict= x_predict.rename(columns={0: 'occupation1', 1: 'occupation2', 2: 'occupation3', 3: 'occupation4', 4: 'occupation5',
    #                       5: 'occupation6', 6: 'occupation_husb1', 7: 'occupation_husb2', 8: 'occupation_husb3',
    #                       9: 'occupation_husb4', 10: 'occupation_husb5', 11: 'occupation_husb6', 12: 'rate_marriage',
    #                       13: 'age', 14: 'yrs_married', 15: 'children', 16: 'religious', 17: 'educ'})


    print('prediction is', pred)

#
#
predict([[3,27,13,3,1,14,3,4]])


