import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms


def k_NN(X_train, X_test, y_train, y_test, k):
    # KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
    # metric_params=None, n_jobs=None)
    knn_classifier = KNeighborsClassifier(n_neighbors = k, algorithm='brute')

    # Train the k-NN classifier and count the training time
    knn_classifier.fit(X_train, y_train)

    # Using trained model, predict the result using test data
    knn_pred = knn_classifier.predict(X_test)
    print(knn_pred)
    print(len(knn_pred))

    wrong = np.array([])
    for i in range(len(knn_pred)):
        if knn_pred[i] != y_test[i]:
            wrong = np.append(wrong, y_test[i])

    clas, freq = np.unique(wrong, return_counts=True)

    # Find the index of the maximum count
    most_frequent_index = np.argmax(freq)

    # Find the most frequent element
    most_frequent_element = clas[most_frequent_index]

    print(most_frequent_element)
    # Get the accuracy for the classifier
    knn_accuracy = accuracy_score(y_test, knn_pred)

    # Print the result
    print("k-NN Accuracy:", knn_accuracy)


if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # transform = transforms.Compose([transforms.ToTensor()])
    # print("abc")
    # mnist_trainset = torchvision.datasets.MNIST(train=True, download=False, transform=transform)
    # print("abc")
    # mnist_testset = torchvision.datasets.MNIST(train=False, download=False, transform=transform)
    #
    # print(len(mnist_trainset))
    # print(len(mnist_testset))

    train_X_2 = train_X.reshape(60000, 784)/256
    test_X_2 = test_X.reshape(10000, 784)/256


    k_NN(train_X_2, test_X_2, train_y, test_y, 1)