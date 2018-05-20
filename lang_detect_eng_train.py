import os
import pickle

import numpy
import pandas
from keras.layers import Embedding, Convolution1D, Dropout, Dense, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# TODO: use n-grams instead of bag of words approach?

SEED = 7
numpy.random.seed(SEED)

# languages that will be used
LANGUAGES = ['eng', 'fra', 'spa', 'deu', 'pol', 'ita']
LANGUAGE_TO_DETECT = 'eng'

# map used to keep consistent language labels between cross language dataset and tatoeba dataset
TATOEBA_LANG_MAP = {'en': 'eng', 'fr': 'fra', 'es': 'spa'}


def load_cross_language_dataset(dataset_path='Cross-Language-Dataset/Wikipedia/') -> pandas.DataFrame:
    dataset = []
    languages = ['en', 'es', 'fr']

    for lang in languages:
        lang_dataset_path = dataset_path + lang
        for dataset_file in os.listdir(lang_dataset_path):
            with open(os.path.join(lang_dataset_path, dataset_file), mode='r', encoding='utf-8') as file:
                text = file.read()
                dataset.append({'lang': TATOEBA_LANG_MAP[lang], 'text': text})
    dataset = pandas.DataFrame(data=dataset)
    return dataset


def load_tatoeba_dataset(langs_to_use: set, dataset_path='Tatoeba-Dataset/sentences.csv') -> pandas.DataFrame:
    dataset = pandas.read_csv(dataset_path, sep='\t')
    dataset.columns = ['id', 'lang', 'text']
    # Remove id column
    dataset.drop(['id'], axis=1)
    # Keep only languages defined in langs_to_use
    dataset = dataset.loc[dataset['lang'].isin(langs_to_use)]
    return dataset

#TODO: get dataset data as script arguments
dataset = load_tatoeba_dataset(set(LANGUAGES))
# Shuffle dataset
dataset = dataset.sample(frac=1)
print(dataset[:10])

# Tokenize texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['text'])
preprocessed_texts = tokenizer.texts_to_sequences(dataset['text'])

max_texts_count = 1000000
preprocessed_texts = preprocessed_texts[:max_texts_count]

sentences_count = len(preprocessed_texts)
print('Sentences count: {}'.format(sentences_count))

max_sentence_length = 200

# Padding
preprocessed_texts = sequence.pad_sequences(preprocessed_texts, maxlen=max_sentence_length)

word_index = tokenizer.word_index
#print(word_index)
# print(preprocessed_texts[:10])

# Dump fitted tokenizer
with open('lang-detect-eng-tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Map languages for binary classification
classification_map = {lang: 0 for lang in LANGUAGES}
classification_map[LANGUAGE_TO_DETECT] = 1
preprocessed_labels = [classification_map[lang] for lang in dataset['lang']]
preprocessed_labels = preprocessed_labels[:max_texts_count]
# Create model
model = Sequential()
model.add(Embedding(len(word_index) + 1, 32, input_length=max_sentence_length))
model.add(Convolution1D(filters=64, kernel_size=4, padding='valid', activation='relu'))
model.add(MaxPooling1D(4))
model.add(Dropout(0.2))
model.add(Convolution1D(filters=32, kernel_size=2, padding='valid', activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Train model
print(preprocessed_texts[:10])
preprocessed_labels = numpy.asarray(preprocessed_labels)
print(preprocessed_labels[:10])

input_dim = preprocessed_texts.shape[1]
print('input dim: {}'.format(input_dim))

model.fit(x=preprocessed_texts, y=preprocessed_labels,
          validation_split=0.3, epochs=1, batch_size=512, verbose=1)
#  Epoch 1/1 700000/700000 [==============================]
#  - 270s - loss: 0.0332 - acc: 0.9829 - val_loss: 0.0036 - val_acc: 0.9989

model.save('lang-detect-eng-model.h5')
