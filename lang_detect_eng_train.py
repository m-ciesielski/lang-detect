import os
import pickle

import numpy
import pandas
from keras.layers import Embedding, Convolution1D, Dropout, Dense, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# TODO: reduce bias to positive output, use n-grams instead of bag of words approach

SEED = 7
numpy.random.seed(SEED)

# Load Wikipedia dataset for training
dataset_path = 'Cross-Language-Dataset/Wikipedia/'

dataset = []
languages = ['en', 'es', 'fr']

for lang in languages:
    lang_dataset_path = dataset_path + lang
    for dataset_file in os.listdir(lang_dataset_path):
        with open(os.path.join(lang_dataset_path, dataset_file), mode='r', encoding='utf-8') as file:
            text = file.read()
            dataset.append({'lang': lang, 'text': text})

# Use pandas for further processing
dataset = pandas.DataFrame(data=dataset)
print(dataset[:10])

# Tokenize texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['text'])
# print(dataset['text'][:10])
preprocessed_texts = tokenizer.texts_to_sequences(dataset['text'])

sentences_count = len(preprocessed_texts)
print('Sentences count: {}'.format(sentences_count))

max_sentence_length = max(len(sentence) for sentence in preprocessed_texts)
print('Max sentence length: {}'.format(max_sentence_length))

# Padding
preprocessed_texts = sequence.pad_sequences(preprocessed_texts, maxlen=max_sentence_length)

word_index = tokenizer.word_index
print(word_index)
# print(preprocessed_texts[:10])

# Dump fitted tokenizer
with open('lang-detect-eng-tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Map languages for binary classification
classification_map = {'en': 1, 'es': 0, 'fr': 0}
preprocessed_labels = [classification_map[lang] for lang in dataset['lang']]
# print(preprocessed_labels[10:])

# Create model
model = Sequential()
model.add(Embedding(len(word_index) + 1, 8, input_length=max_sentence_length))
# TODO: SpatialDropout?
model.add(Convolution1D(filters=16, kernel_size=3, padding='valid', activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Train model
#preprocessed_texts = numpy.asarray(preprocessed_texts)
print(preprocessed_texts[:10])
preprocessed_labels = numpy.asarray(preprocessed_labels)
print(preprocessed_labels[:10])

input_dim = preprocessed_texts.shape[1]
print('input dim: {}'.format(input_dim))

model.fit(x=preprocessed_texts, y=preprocessed_labels,
          validation_split=0.4, epochs=6, batch_size=32, verbose=2)

test_sentences = ['Ich bin Hitler!',
                  'I like trains',
                  'How it is going? Well',
                  'Omelette du fromage',
                  'Grzegorz Brzeczyszczykiewicz',
                  'Chocolate milk is so much better through a straw. I lack said straw',
                  'Pobrać fotokod biletu z linku wysłanego do Ciebie SMS-em - wystarczy połączenie z Internetem (stawka za połączenie zależy od Twojej taryfy).',
                  'Abonnez-vous à la chaine You Tube pour ne manquer aucune vidéo : cliquez ici. Téléchargez le fichier MP3 ici. Téléchargez le fichier PDF ici.']

preprocessed_test_sentences = tokenizer.texts_to_sequences(test_sentences)
preprocessed_test_sentences = sequence.pad_sequences(preprocessed_test_sentences, maxlen=max_sentence_length)

predictions = model.predict(preprocessed_test_sentences)
for i, p in enumerate(predictions):
    print(test_sentences[i])
    print(p)