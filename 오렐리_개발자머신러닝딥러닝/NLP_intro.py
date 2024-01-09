import tensorflow as tf
from keras.preprocessing.text import Tokenizer

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

#print(f'seq : {sequences}')

test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

test_sequence = tokenizer.texts_to_sequences(test_data)
#print(word_index)
#print(test_sequence)

# 패딩
from keras.preprocessing.sequence import pad_sequences
dif_sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed working in the snow today'
]

tokenizer.fit_on_texts(dif_sentences)
new_sequences = tokenizer.texts_to_sequences(dif_sentences)
#print(new_sequences)

padded = pad_sequences(new_sequences)
#print(f'padded: {padded}')

from bs4 import BeautifulSoup
soup = BeautifulSoup(sentences)
sentence = soup.get_text()

stopwords = ["a", "about", "above", "yours", "yourself", "yourselves"]

words = sentence.split()
