import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

t = np.arange(0, 1000)
x = np.sin(0.02*t) + np.random.rand(1000)
train, test = x[0:800], x[800:1000]

def convert_to_matrix(data, step):
    x, y = [], []
    for i in range(len(data)-step):
        d = i+step
        x.append(data[i:d])
        y.append(data[d])
    return np.array(x), np.array(y)

train = np.append(train, np.repeat(train[-1], 4))
test = np.append(test, np.repeat(test[-1], 4))

train_x, train_y = convert_to_matrix(train, 4)
test_x, test_y = convert_to_matrix(test, 4)

model = Sequential()
model.add(SimpleRNN)




