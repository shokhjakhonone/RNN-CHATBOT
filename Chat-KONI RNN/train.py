import numpy as np 
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM, Attention, LayerNormalization, GlobalAveragePooling1D, Dropout, SpatialDropout1D
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from keras_tuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import pickle
import json
import nltk
from keras.layers import Attention
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load stop words
stop_words = set(stopwords.words('russian'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
with open('intents4.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Пример ваших вопросов
questions = [intent['question'] for intent in data['intents']]

# Создание TF-IDF векторайзера
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(questions)

# Получение ключевых слов для каждого вопроса
feature_names = vectorizer.get_feature_names_out()

# Сортировка индексов слов по весам TF-IDF и выбор топ N ключевых слов
top_n = 5
keywords_list = []

for i in range(X_tfidf.shape[0]):
    indices = X_tfidf[i].indices
    sorted_indices = sorted(indices, key=lambda x: X_tfidf[i, x], reverse=True)
    keywords = [feature_names[idx] for idx in sorted_indices[:top_n]]
    keywords_list.append(keywords)

# Добавьте ключевые слова к вашим данным
for i, intent in enumerate(data['intents']):
    intent['keywords'] = keywords_list[i]

# Предобработка текста с использованием ключевых слов
def preprocess_text_with_keywords(text, keywords):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    words.extend(keywords)
    return ' '.join(words)

# Преобразуйте ваши вопросы, добавив ключевые слова
training_sentences = [preprocess_text_with_keywords(intent['question'], intent['keywords']) for intent in data['intents']]



training_sentences = []
training_labels = []
labels = []
responses = []

# Preprocess the text
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub('[^а-яА-Я]', ' ', text)
    
    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Lemmatize and remove stop words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    
    return ' '.join(words)

for intent in data['intents']:
    if len(intent['tags']) > 0:
        training_sentences.append(preprocess_text(intent['question']))
        training_labels.append(intent['tags'][0])
        responses.append(intent['answer'])
        
        if intent['tags'][0] not in labels:
            labels.append(intent['tags'][0])
        
num_classes = len(labels)
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
vocab_size = 1000
embedding_dim = 64
max_len = 100
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token) # adding out of vocabulary token
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Define the model builder function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))

    for i in range(hp.Int('num_bidirectional_layers', 1, 6)):
        model.add(Bidirectional(LSTM(units=hp.Int('units_lstm_' + str(i), min_value=50, max_value=150, step=10),
                                     return_sequences=True,
                                     dropout=0.2,
                                     recurrent_dropout=0.2)))
        
        # Различные типы слоев внимания: AdditiveAttention, MultiHeadAttention
        if i % 2 == 0:
            model.add(Attention(use_scale=True))
        else:
            model.add(MultiHeadAttention(num_heads=hp.Int('num_heads_' + str(i), min_value=1, max_value=8),
                                         key_dim=hp.Int('key_dim_' + str(i), min_value=16, max_value=128),
                                         dropout=0.1))
        
        model.add(LayerNormalization(epsilon=1e-6))

    # Добавим SpatialDropout1D
    model.add(SpatialDropout1D(0.2))

    # Добавим GlobalAveragePooling1D для усреднения по времени
    model.add(GlobalAveragePooling1D())

    # Добавим Dropout для регуляризации
    model.add(Dropout(0.5))

    for i in range(hp.Int('num_dense_layers', 1, 20)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=16, max_value=256, step=16), activation='relu'))
        model.add(LayerNormalization(epsilon=1e-6))
        model.add(Dropout(0.3))  # Добавим Dropout перед каждым Dense слоем для регуляризации

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Define the tuner
# tuner = RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=5,
#     executions_per_trial=1,
#     directory='tuner_directory',
#     project_name='ai_chatbot'
# )

# Perform hyperparameter tuning
# tuner.search(padded_sequences, np.array(training_labels), epochs=35, validation_split=0.2)

# Get the best model
# best_model = tuner.get_best_models(num_models=1)[0]

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(LSTM(110, return_sequences=True))
model.add(LSTM(110))
model.add(Dense(208, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 150

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(padded_sequences, training_labels, test_size=0.2, random_state=42)

# Train the model on the training set
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# Evaluate the model on the testing set
# test_loss, test_accuracy = model.evaluate(X_test, np.array(y_test))
# print("Test Loss:", test_loss)
# print("Test Accuracy:", test_accuracy)

# saving model
model.save("chat_koni_model")

# saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# saving label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

