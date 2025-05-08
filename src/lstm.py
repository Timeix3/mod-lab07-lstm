import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
import random
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

words = text.split() 

vocabulary = sorted(list(set(words)))
word_to_idx = dict((word, i) for i, word in enumerate(vocabulary))
idx_to_word = dict((i, word) for i, word in enumerate(vocabulary))

max_length = 3
step = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_length, step):
    sentences.append(words[i: i + max_length])
    next_words.append(words[i + max_length])

X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_idx[word]] = 1
    y[i, word_to_idx[next_words[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample_word(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

model.fit(X, y, batch_size=128, epochs=50)

def generate_text(word_count, diversity):
    start_index = random.randint(0, len(words) - max_length - 1)
    generated = []
    sequence = words[start_index: start_index + max_length]
    generated.extend(sequence)
    
    for i in range(word_count):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, word in enumerate(sequence):
            x_pred[0, t, word_to_idx[word]] = 1
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample_word(preds, diversity)
        next_word = idx_to_word[next_idx]
        
        generated.append(next_word)
        sequence = sequence[1:] + [next_word]

    return ' '.join(generated)

with open('../result/gen.txt', 'w', encoding='utf-8') as file:
    file.write(generate_text(1000, 0.5)) 