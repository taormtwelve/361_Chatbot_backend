import csv
import numpy as np
import deepcut
from keras.models import Model
from keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, Dropout, Concatenate
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
import keras
from pythainlp import sent_tokenize, word_vector
import matplotlib as mpl
from keras.callbacks import CSVLogger


def prepare_data(file_name):
    labels = []
    sentences = []
    file = open(file_name, 'r', encoding='utf-8')
    data = list(csv.reader(file))
    shuffle(data)
    for d in data:
        try:
            labels.append(int(d[0]))
        except:
            labels.append(0)
        sentences.append(d[1].replace(' ',''))
    return labels, sentences


def sent2vec(sentences):
    words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]
    max_sentence_length = max([len(s) for s in words])
    word_vectors = np.zeros((len(words),max_sentence_length,word_vector_length))
    sample_count = 0
    for s in words:
        word_count = 0
        for w in s[::-1]:
            try:
                word_vectors[sample_count,max_sentence_length-word_count-1,:] = wvmodel[w] # wvmodel_pythainlp
                word_count = word_count+1
            except:
                pass
        sample_count = sample_count+1
    return word_vectors, max_sentence_length


def get_model(hidden_nodes, batch_size, epochs, model_count):
    inputLayer = Input(shape=(max_sentence_length,word_vector_length))
    h1 = LSTM(hidden_nodes, activation='relu', return_sequences=True)(inputLayer)
    h1 = Dropout(0.1)(h1)
    h1 = LSTM(hidden_nodes, activation='relu')(inputLayer)
    h1 = Dropout(0.5)(h1)
    outputLayer = Dense(11, activation='softmax')(h1)
    model = Model(inputs=inputLayer, outputs=outputLayer)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    # Create check point for saving model
    checkpoint = ModelCheckpoint(f'./models/chatbot_best_val_{model_count}.h5',
                                 verbose=0,
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max')
    history = model.fit(word_vectors[:160], 
                        to_categorical(labels[:160]), 
                        epochs=epochs, batch_size=batch_size, 
                        validation_split = 0.2,
                        verbose=0,
                        callbacks=[checkpoint])
    return model,history


word_vector_length = 300
wvmodel = word_vector.get_model()
model_count = 3

# for _ in range(15):
#     model_count +=1
#     labels, sentences = prepare_data('./corpus/chatbot_corpus_classification.csv')
#     word_vectors, max_sentence_length = sent2vec(sentences)
#     model, history = get_model(hidden_nodes=128,batch_size=32,epochs=1000, model_count=model_count)
#     max_val = max(history.history['val_accuracy'])
#     np.save(f'./models/model_history_{model_count}_{max_val}.npy',history.history)
#     print('model', model_count , ':', max(history.history['val_accuracy']))
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.show()


# y_pred = model.predict(word_vectors[:])
# cm = confusion_matrix(labels[:], y_pred.argmax(axis=1))
# print('Confusion Matrix')
# print(cm)
# print(max(history.history['val_accuracy']))
