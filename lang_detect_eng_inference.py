import pickle

import pandas
from keras.preprocessing import sequence
from keras.models import load_model

# TODO: parametrize
model_path = 'lang-detect-eng-model-v3.h5'
tokenizer_path = 'lang-detect-eng-tokenizer.pkl'
test_dataset_path = 'Twitter-Test-Dataset/joy-1.csv'

model = load_model(model_path)

with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

tweet_dataset = pandas.read_csv(test_dataset_path, delimiter=' ', quotechar='|')

preprocessed_tweets = tokenizer.texts_to_sequences(tweet_dataset['Text'])
max_sentence_length = 1960
preprocessed_tweets = sequence.pad_sequences(preprocessed_tweets, maxlen=max_sentence_length)

predictions = model.predict(preprocessed_tweets, verbose=2)
positive_pred_count = 0
for i, p in enumerate(predictions):
    if p > 0.6:
        print(tweet_dataset['Text'][i])
        print(p)
        positive_pred_count += 1

print('All sentences: {}'.format(len(preprocessed_tweets)))
print('Classified as positive: {}'.format(positive_pred_count))
