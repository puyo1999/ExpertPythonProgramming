import autokeras as ak
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

def main():
    '''Main method'''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)

    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(x_train[1234])
    ax.set_title('train sample')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(x_test[1234])
    ax.set_title('test sample')
    plt.show()
    '''
    train_histogram = np.histogram(y_train)
    test_histogram = np.histogram(y_test)
    _, axs = plt.subplots(1, 2)

    axs[0].set_xticks(range(1,2))
    axs[0].bar(range(10),train_histogram[0])
    axs[0].set_title('training dataset histogram')

    axs[1].set_xticks(range(1,2))
    axs[1].bar(range(10),test_histogram[0])
    axs[1].set_title('test dataset histogram')
    plt.show()

def main2():
    ar = np.array([[0,1,2,3,4],[5,6,7,8,9]])

    #print(ar[:,-1])
    print(np.max(ar, axis=1))
def main3():
    arr = np.arange(0, 4*2*4)
    print("arr {}".format(arr))
    v = arr.reshape([4,2,4])
    print("v {}".format(v))
    print("v dim {}".format(v.ndim))
    print("v sum {}".format(v.sum()))
    res01 = v.sum(axis=0)
    print("res01 shape {}\nres01\n{}".format(res01.shape, res01))

    res02 = v.sum(axis=1)
    print("res02 shape {}\nres02\n{}".format(res02.shape, res02))

    res03 = v.sum(axis=2)
    print("res03 shape {}\nres03\n{}".format(res03.shape, res03))



if __name__ == '__main__':
    #main()
    #main2()
    #main3()