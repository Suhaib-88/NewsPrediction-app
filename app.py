import numpy as np
from flask import Flask,render_template, request 
import spacy
import os
import pickle 

from newsapi import NewsApiClient


os.putenv('LANG', 'en_US.UTF-8')

app= Flask(__name__)

model= pickle.load(open("predModel.pkl", 'rb'))


@app.route('/',methods=['GET'])                                                                                                                                           
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        texts=request.form["message"]
        prediction= model.predict_proba(np.array([texts]))
        pred_val=np.round(np.max(prediction),2)
        max_val=np.argmax(prediction)
        target_category=model.classes_[max_val]

        def show_result():
            a=dict(enumerate(model.classes_.flatten()))
            b=dict(enumerate(np.round(prediction,2).flatten()))
            ds = [a, b]
            d = {}
            for k in a.keys():
                d[k] = tuple(d[k] for d in ds)
            return d.values()

        def show_labels(x):
            ner=spacy.load("en_core_web_sm")
            doc=ner(x).ents
            a=[(tag.text,tag.label_) for tag in doc]
            return a

        label= show_labels(request.form['message'])
        result=show_result()
        
        with open('password.txt','r') as file:
            key=file.read()
        
        api=NewsApiClient(api_key=key)
        data=api.get_everything(q=request.form['message'],language='en')
        
        try:
            urls=data['articles'][0]['url']
            date=data['articles'][0]['publishedAt'][:10]
        except:
            pass
        
        return render_template('result.html',prediction=result,target=target_category,pred=pred_val,label=label,date=date,urls=urls)
    
    

@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')   

if __name__=="__main__":
    app.run()
