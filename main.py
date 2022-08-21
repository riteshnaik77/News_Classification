# importing libraries
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import flask
import pickle
from flask import Flask, render_template, request

# creating instance of the class
app = Flask(__name__)


# to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('news-home.html')


# prediction function
def ValuePredictor(to_predict_list):
    loaded_model = pickle.load(open("news_classification.pkl", "rb"))
    load_tfidf = pickle.load(open("news_classification_tfidf_vectorizer.pkl", "rb"))

    test_article = to_predict_list.lower()
    test_frame = pd.DataFrame({"Text": [test_article]})
    print(test_frame)

    test_feature = load_tfidf.transform(test_frame.Text).toarray()
    print("Checking this", test_feature)

    prediction = loaded_model.predict(test_feature)
    id_to_category = {0: "business", 1: "tech", 2: "politics", 3: "sport", 4: "entertainment"}
    pred_category = id_to_category[prediction[0]]

    return pred_category

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form["text"]
        print(to_predict_list)
        result = ValuePredictor(to_predict_list)

        return render_template("news-cat.html", prediction=f"Model predicts this excerpt belong to {str.upper(result)} category!")







if __name__ == "__main__":
    app.run(debug=True)