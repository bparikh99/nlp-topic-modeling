from flask import Flask ,request, jsonify, render_template
import pickle
import numpy as np
from cloud.word_cloud import wordtoplot
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os import path
import time
# from common import TestView

app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    doc=str(request.form['txarea'])
    # print(doc)
    with app.test_request_context():
        obj=wordtoplot(doc)
        obj.clean_text()
        obj.do_lemma()
        obj.make_df()
        obj.word_plot()
        obj.common_plot()
        topics=obj.get_topics()
    return render_template('predict.html',topics=topics)

# def predict2():
#     with app.test_request_context():
#         obj=TestView()
#         topics=obj.get()
#     return render_template('predict.html',topics=topics)

if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='localhost', debug=True)