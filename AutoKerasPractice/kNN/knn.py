import numpy as np
from sklearn import datasets

def random_sampling(high, n_size):
    result = []
    rand_num = np.random.randint(high)
    for _ in range(n_size):
        while rand_num in result:
            rand_num = np.random.randint(high)
        result.append(rand_num)
    return np.array(result)

def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

iris = datasets.load_iris()
X = iris.data[:,2:]
y = iris.target

# random Sampling & array 기반 인덱싱
test_idx = random_sampling(150, 30)
train_idx = np.ones(len(X), dtype=bool)
train_idx[test_idx] = False

# Learning data 추출
X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("X_train: %s y_train: %s X_test: %s y_test: %s" % (len(X_train),len(y_train),len(X_test),len(y_test)))

class kNN():
    # 초기에 k 값을 지정
    def __init__(self,k=3):
        self.k = k

    # Train: Train데이터 모두 memory에 저장
    def train(self, X, y):
        self.X = X
        self.y = y

    def get_distance(self, X_test, method="L2"):
        N = len(X_test)
        M = len(self.X)
        X_train_t = np.tile(self.X, (N,1))
        X_test_t = np.repeat(X_test, M, axis=0)

        if method == "L2":
            distance = np.sqrt(np.sum((X_train_t - X_test_t) ** 2, axis=1)).reshape(N, M)
        return distance

    # 다수결 투표를 통한 label 도출
    def predict(self, X_test):
        N = len(X_test)
        y_hat = np.zeros(N)

        distance = self.get_distance(X_test, method="L2")
        arg_dist = distance.argsort()
        for i in range(N):
            row = arg_dist[i]
            k_neighbor = self.y[row[:self.k]]
            target, cnt = np.unique(k_neighbor, return_counts=True)
            y_hat[i] = target[np.argmax(cnt)]
        return y_hat

model = kNN(k=3)
model.train(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy(y_pred, y_test)
print("Accuracy: ", acc)

import matplotlib.pyplot as plt

test_point = np.array([[2.5, 0.5], [5,1], [3,2],])

test_pred=  model.predict(test_point)
test_pred = list(map(int, test_pred))

cdict = {0: 'red', 1: 'blue', 2: 'green'}
fig, ax = plt.subplots()
for g in np.unique(y_train):
    ix = np.where(y_train == g)
    ax.scatter(X_train[:,0][ix], X_train[:,1][ix], c = cdict[g], label = g)
for g in np.unique(test_pred):
    ix = np.where(test_pred == g)
    ax.scatter(test_point[:,0][ix], test_point[:,1][ix], c = cdict[g], label = 'test_'+str(g), marker="D",s=100)

ax.legend()
plt.show()