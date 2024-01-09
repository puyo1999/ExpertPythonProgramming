from keras import layers
from keras import Input
from keras.models import Model
vocab_size = 50000
num_inc_group= 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(vocab_size, 256)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)

income_prediction = layers.Dense(num_inc_group, activtion='softmax', name='income')(x)

model = Model(posts_input, [age_prediction, income_prediction])
model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
# 혹은 층에 일므을 지정했으면, 아래와 같이도 사용



