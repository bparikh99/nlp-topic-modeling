from flask import Flask ,request, jsonify, render_template
import pickle
import numpy as np
from cloud.word_cloud import wordtoplot


app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    doc=str(request.form['txarea'])
    # print(doc)
    with app.test_request_context():
        obj=wordtoplot(doc)
        obj.clean_text()
        obj.do_lemma()
        obj.make_df()
        obj.word_plot()
    return render_template('predict.html',name = 'Documented Image', url ='/static/images/cloud.png')


if __name__ == "__main__":
    app.run(host='localhost', debug=True)
