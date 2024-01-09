import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))


from bs4 import BeautifulSoup





