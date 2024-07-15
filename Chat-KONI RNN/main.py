import json
import numpy as np
from tensorflow import keras
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import time
from flask import Flask, render_template, request, jsonify

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    stop_words = set(stopwords.words('russian'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub('[^а-яА-Я]', ' ', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

def get_random_response(intent):
    responses = intent['answer'].split('|')
    return np.random.choice(responses)

def is_clear(question):
    # Добавьте логику для определения ясности вопроса
    # Возможно, используя какие-то эвристики или алгоритмы
    return True

def respond_to_question(question):
    preprocessed_input = preprocess_text(question)
    sequence = tokenizer.texts_to_sequences([preprocessed_input])
    padded_sequence = keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', maxlen=max_len)
    result = model.predict(padded_sequence)
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    probability = np.max(result)

    response = ""

    # Проверяем, совпадает ли вопрос с каким-либо вопросом из intents4.json
    question_matched = any(preprocess_text(intent['question']) == preprocessed_input for intent in data['intents'])

    if not question_matched:
       responses_not_matched = [
        "Я ещё в стадии обучения. Юридические термины пока для меня как абракадабра, но задай свой вопрос, и мы вместе попробуем разгадать этот код!",
       
    ]
       response = np.random.choice(responses_not_matched)
    else:
        for i in data['intents']:
            if len(i['tags']) > 0:
                if i['tags'][0] == tag:
                    response = get_random_response(i)
                    if tag not in ["greeting", "goodbye", "thanks", "salom_tong", "nima_gap", "raxmat_kotta", "sarson_yuristlik"]:
                        response = f"Это интересный вопрос, вот что я думаю: {response}"
                    break

        if not is_clear(question):
            response = "Ваш вопрос не совсем ясен, но я постараюсь ответить."

    # Добавьте логику для вставки фраз-заполнителей, вопросов в ответе и выражения уверенности

    return response, probability

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    inp = request.form['user_input']

    response, probability = respond_to_question(inp)

    for word in response.split():
        time.sleep(0.008)  # Задержка в десятую секунды между выводом слов
        print(word, end=' ', flush=True)

    probability = float(probability)  # Преобразование в обычный float

    # Сохранение вопроса пользователя в JSON файле
    user_question_data = {
        "user_question": inp
    }
    with open('user_questions.json', 'a', encoding='utf-8') as json_file:
        json.dump(user_question_data, json_file, ensure_ascii=False)
        json_file.write('\n')

    return jsonify({'response': response, 'probability': probability})

if __name__ == '__main__':
    with open('intents.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    model = keras.models.load_model('chat_koni_model')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    max_len = 100

    app.run(debug=True, port=80)

