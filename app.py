# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request
import pickle

app = Flask(__name__)
#loaded_model = pickle.load(open(filename, 'rb'))


@app.route("/")
def hello():
    return render_template("tags_prediction.html")

@app.route("/api/tags_prediction")
def tags_prediction():
    #question = ''
    #result = loaded_model.score(X_test, y_test)
    #print(result)
    title = request.args.get("title")
    question = request.args.get("question")
    return jsonify({"title":title, "question":question})


def pre_processing():
    # preprocessing the question
    return
    
if __name__ == "__main__":
    app.run(debug=True)