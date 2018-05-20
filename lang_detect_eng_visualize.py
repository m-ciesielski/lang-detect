import pickle

import matplotlib.pyplot as plt
import pandas
from keras.preprocessing import sequence
from keras.models import load_model
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

model_path = 'lang-detect-eng-model-v3.h5'
tokenizer_path = 'lang-detect-eng-tokenizer.pkl'
test_dataset_path = 'Twitter-Test-Dataset/joy-1.csv'
model = load_model(model_path)

with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

tweet_dataset = pandas.read_csv(test_dataset_path, delimiter=' ', quotechar='|')

preprocessed_tweets = tokenizer.texts_to_sequences(tweet_dataset['Text'])
max_sentence_length = 200
preprocessed_tweets = sequence.pad_sequences(preprocessed_tweets, maxlen=max_sentence_length)

# layer_idx = utils.find_layer_idx(model, 'preds')
layer_idx = -1

# Swap sigmoid with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# Pick some sentence
input_sentence = preprocessed_tweets[16]
input_sentence_raw = tweet_dataset['Text'][16]

grads = visualize_saliency(model, layer_idx, filter_indices=None, seed_input=input_sentence, wrt_tensor=model.layers[0].output)

# Cut padding from gradients
words = input_sentence_raw.split(' ')
# Remove pre-padding
word_grads = grads[-(len(words)):]

x = [i for i in range(len(words))]
plt.bar(x, word_grads)
plt.xticks(x, words)
plt.show()
