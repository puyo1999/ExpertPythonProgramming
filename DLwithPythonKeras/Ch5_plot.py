import os
from keras import layers
from keras import models

import matplotlib.pyplot as plt
import keras.models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

base_dir = './datasets/cats_and_dogs_smaall'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))

model.summary()
model.compile(loss='binary_crossentropy', # sigmoid 이므로 binary 사용
              optimizer=optimizers.RMSProp(lr=1e-4),
              metrics=['acc']) # 정확도

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size = 20,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b',label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.show()