# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request
import joblib
import pandas as pd
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import re

app = Flask(__name__)

model = joblib.load('final_model.pkl')
target_name = pd.read_csv('target_names.csv')
clean_words_post = pd.read_csv('clean_words_post.csv')
print(model)


@app.route("/")
def hello():
    return render_template("tags_prediction.html")

@app.route("/api/tags_prediction")
def tags_prediction():
    
    
    title = request.args.get("title")
    question = request.args.get("question")
    df = pd.DataFrame({'title':[title], 'body':[question]})
    df_preproc = pre_process(df)
    pred = model.predict(df_preproc)
    tags = get_tags_name(pred)
    
    return jsonify(tags)


def get_tags_name(y):
    
    # unique passage car on prédit un seul post à la fois
    tag_pred = ' / '.join([target_names.iloc[i] for i,v in enumerate(y[0]) if v == 1])
    
    if tag_pred == '':
        tag_pred = 'There is no suggested tag for your question !'
    
    return tag_pred

def pre_process(df):
    
    title = df.title.copy()

    # On retire la ponctuation et on réduit la casse
    title = title.map(pre_tokenize)

    # On retire les caractères n'étant pas des lettres
    title = title.map(lambda x: re.sub("[^a-zA-Z]", " ", x))

    # On garde la racine des mots complexes
    title = title.map(lemmatize_text)
    
    body = df.body.copy()
    
    # On parse le texte pour retirer les balises html et conserver le texte
    body = body.map(lambda text: get_bs_text(text))

    # On retire les caractères n'étant pas des lettres
    body = body.map(lambda x: re.sub("[^a-zA-Z]", " ", x))

    # On retire la ponctuation et on réduit la casse
    body = body.map(pre_tokenize)

    # On garde la racine des mots complexes
    body = body.map(lemmatize_text)
    
    X_new = pd.DataFrame({'title':title, 'body':body})
        
    X_new = X_new['title']+X_new['body']
    
    X_new = X_new.map(lambda x: [word for word in x if word in list(clean_words_post)])
    
    X_new = vectorizer.transform(X_new.map(lambda x: ' '.join(x)))

    return X_new

def pre_tokenize(text):
    """ Permet de supprimer la ponctuation et lowercase """
    
    # Les ponctuations à supprimer
    expr = ["?", ",", ".", "'", ";", ":", "!", "\n", "(", ")", "|", "_", "-", "`", "-", "*", "\""]
    for char in expr:
        text = text.replace(char, ' ')
    # Réduire la casse   
    text = str(np.char.lower(text))
    
    return text

def lemmatize_text(text):
    """ Permet de retirer prefixe et suffixe pour retenir la racine du mot """

    lem = WordNetLemmatizer()
    text_lemmatized = []
    words = word_tokenize(text)
    for word in words:
        word = lem.lemmatize(word, "n")
        word = lem.lemmatize(word, "v")
        text_lemmatized.append(word)
    #text = ' '.join(text_lemmatized)
    
    return text_lemmatized

def get_bs_text(raw_text):
    soup = BeautifulSoup(raw_text)
    divTag = soup.find_all("p")

    clean_text = ' '.join([elem.text for elem in divTag])
    return clean_text

    
if __name__ == "__main__":
    app.run(debug=True)