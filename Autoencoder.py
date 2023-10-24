#2023.10.20
#Implement autoencode with knn

from keras.datasets import mnist
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cityblock
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA as skl_PCA
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Flatten,Reshape, LeakyReLU as LR, Activation, Dropout
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D


def autoencoder(layers_size, dataset):
    # encoder function
    encoder = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(layers_size[0]),
        LR(),
        Dropout(0.5),
        Dense(layers_size[1]),
        LR(),
        Dropout(0.5),
        Dense(layers_size[2]),
        LR(),
        Dropout(0.5),
        Dense(layers_size[3]),
        LR(),
        Dropout(0.5),
        Dense(layers_size[4]),
        LR(),
        Dropout(0.5),
        Dense(layers_size[5]),
        LR()
    ])

    # decoder function
    # decoder = Sequential([
    #     Dense(layers_size[4], input_shape=(layers_size[5],)),
    #     LR(),
    #     Dropout(0.5),
    #     Dense(layers_size[3]),
    #     LR(),
    #     Dropout(0.5),
    #     Dense(layers_size[2]),
    #     LR(),
    #     Dropout(0.5),
    #     Dense(layers_size[1]),
    #     LR(),
    #     Dropout(0.5),
    #     Dense(layers_size[0]),
    #     LR(),
    #     Dropout(0.5),
    #     Dense(784),
    #     Activation("sigmoid"),
    #     Reshape((28, 28))
    # ])

    img = Input(shape=(28, 28))

    # Create the encoder and decoder
    latent_vector = encoder(img)
    model_encoder = Model(img, latent_vector)
    # output = decoder(latent_vector)
    # model = Model(inputs=img, outputs=output)
    # model.compile("nadam", loss="binary_crossentropy")
    #
    # # Train the data using itself as the y too
    # history = model.fit(x_train, x_train, epochs = 5)
    # print(history.history['loss'])

    code = model_encoder.predict(dataset)

    return code

def con_autoencoder(layers_size, dataset):
    input_img = Input(shape=(28, 28, 1))  # Assuming grayscale images

    # Encoder
    x = Conv2D(layers_size[0], (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(layers_size[1], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(layers_size[2], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(layers_size[3], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(layers_size[4], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(layers_size[5], (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    print("encoded shape: ", encoded.shape)

    # # Decoder
    # x = Conv2D(layers_size[5], (3, 3), activation='relu', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(layers_size[4], (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(layers_size[3], (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(layers_size[2], (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(layers_size[1], (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # decoded = Conv2D(layers_size[0], (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, encoded)

    # autoencoder.compile(optimizer='nadam', loss='binary_crossentropy')
    #
    # # Assuming dataset is properly preprocessed
    # history = autoencoder.fit(dataset, dataset, epochs=5)  # Adjust the number of epochs as needed
    code = autoencoder.predict(dataset)
    code = code.reshape(dataset.shape[0], layers_size[5])
    print("code shape: ", code.shape)

    return code

def k_NN(X_train, X_test, y_train, y_test, k):
    # KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
    # metric_params=None, n_jobs=None)
    knn_classifier = KNeighborsClassifier(n_neighbors=k, algorithm='brute')

    # Train the k-NN classifier and count the training time
    start_time = 0
    start_time = time.time()
    knn_classifier.fit(X_train, y_train)
    # Using trained model, predict the result using test data
    knn_pred = knn_classifier.predict(X_test)

    end_time = time.time()
    knn_train_time = end_time - start_time

    # Get the accuracy for the classifier
    knn_accuracy = accuracy_score(y_test, knn_pred)

    # Print the result
    # print("k-NN Accuracy:", knn_accuracy)
    # print(f"k-NN Training Time: {knn_train_time:.5f} seconds")

    return knn_accuracy, knn_train_time

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X_2 = train_X.reshape(60000, 784)
    test_X_2 = test_X.reshape(10000, 784)

    x_train = train_X / 255.0
    x_test = test_X / 255.0


    # layers_size = [512, 256, 128, 75, 32, 16]

    # layers_size = [640, 520, 400, 280, 150]
    numOfcom = 5
    maxNumCom = 100
    ks = np.array([1, 3, 5])

    str_array = ks.astype(str)
    col_name = np.insert(str_array, 0, "size of code - k")

    # Create a table
    table = PrettyTable()
    table.field_names = col_name

    while numOfcom <= maxNumCom :
        layers_size = [640, 520, 400, 280, 150]
        layers_size.append(numOfcom)
        row = [numOfcom]
        for k in ks:

            # ---------------------- my autoencoder ----------------------
            coded_train_x = autoencoder(layers_size, x_train)
            coded_test_x = autoencoder(layers_size, x_test)

            my_acc, my_time = k_NN(coded_train_x, coded_test_x, train_y, test_y, k)

            row.append(("(" + str(my_acc) + ", " + str(my_time) + ")"))
            print(numOfcom, " --- ", k)
            print("(" + str(my_acc) + ", " + str(my_time) + ")")
        table.add_row(row)
        numOfcom += 5

    print("Autoencoder with layers_size = [640, 520, 400, 280, 150, 'size of code']: ")
    print(table)

    print("done in main")
    # EPOCHS = 5
    # for epoch in range(EPOCHS):
    #     fig, axs = plt.subplots(4, 4)
    #     rand = x_test[np.random.randint(0, 10000, 16)].reshape((4, 4, 1, 28, 28))
    #
    #     # display.clear_output()  # If you imported display from IPython
    #
    #     for i in range(4):
    #         for j in range(4):
    #             axs[i, j].imshow(model.predict(rand[i, j])[0], cmap="gray")
    #             axs[i, j].axis("off")
    #
    #     plt.subplots_adjust(wspace=0, hspace=0)
    #     plt.show()
    #     print("-----------", "EPOCH", epoch, "-----------")
    #     model.fit(x_train, x_train)













