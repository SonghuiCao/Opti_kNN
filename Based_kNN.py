# 2023.9.26
# Implementation of kNN for Mnist data collection
# 60k training data, and 10k testing data
# Find the digit with the biggest difficulty classify

from keras.datasets import mnist
from scipy.spatial.distance import cityblock
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA as skl_PCA
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time



def distance(x, y):
    # flatten() makes 2d array into 1d array, then do the distance calculation
    return np.linalg.norm(np.array(x.flatten()) - np.array(y.flatten()))

def distance1(x, y):
    # flatten() makes 2d array into 1d array, then do the distance calculation
    return np.linalg.norm(x - y)

def distance2(x, y):
    return np.sqrt(np.sum(np.square(np.array(x.flatten())-np.array(y.flatten()))))

def distance3(x, y):
    return cityblock(np.array(x.flatten()), np.array(y.flatten()))

def kNN(x, k, data, label):
    #create a list of distances between the given image and the images of the training set
    # print("knn test: ",len(x))
    # print("knn train: ", len(data[0]))

    # distances = np.array([])
    # for i in range(len(data)):
    #     # print("knn test: ", len(x))
    #     # print("knn train", i, ": ", len(data[i]))
    #     Adis = distance3(x, data[i])
    #     distances = np.append(distances, Adis)

    distances =[distance(x, data[i]) for i in range(len(data))]
    # Use "np.argpartition". It does not sort the entire array.
    # It only guarantees that the kth element is in sorted position
    # and all smaller elements will be moved before it.
    # Thus the first k elements will be the k-smallest elements.
    idx = np.argpartition(distances, k)
    # print("idx: ", idx)
    # print('idx shape: ' + str(idx.shape))
    clas, freq = np.unique(label[idx[:k]], return_counts=True)

    # print("\nclas: ", clas)
    # print("freq: ", freq)
    return clas[np.argmax(freq)]

def k_NN(X_train, X_test, y_train, y_test, k):
    # KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
    # metric_params=None, n_jobs=None)
    knn_classifier = KNeighborsClassifier(n_neighbors = k, algorithm='brute')

    # Train the k-NN classifier and count the training time
    start_time = 0
    start_time = time.time()
    knn_classifier.fit(X_train, y_train)
    end_time = time.time()
    knn_train_time = end_time - start_time

    # Using trained model, predict the result using test data
    knn_pred = knn_classifier.predict(X_test)
    # print(knn_pred)
    # print(len(knn_pred))

    # wrong = np.array([])
    # for i in range(len(knn_pred)):
    #     if knn_pred[i] != y_test[i]:
    #         wrong = np.append(wrong, y_test[i])
    #
    # clas, freq = np.unique(wrong, return_counts=True)
    #
    # # Find the index of the maximum count
    # most_frequent_index = np.argmax(freq)
    #
    # # Find the most frequent element
    # most_frequent_element = clas[most_frequent_index]
    #
    # print(most_frequent_element)

    # Get the accuracy for the classifier
    knn_accuracy = accuracy_score(y_test, knn_pred)

    # Print the result
    print("k-NN Accuracy:", knn_accuracy)
    print(f"k-NN Training Time: {knn_train_time:.5f} seconds")

def recon_ppt(OrData, transfData, eigvect, mean):
    transpEigvec = np.transpose(eigvect)
    print("shape of transfData:", transfData.shape)
    print("shape of transpEigvec:", transpEigvec.shape)
    AdData = transfData @ transpEigvec
    reconData = AdData + mean
    result = np.allclose(OrData, reconData) | np.array_equal(OrData, reconData)
    diff = OrData - reconData
    print("shape of OrData:", OrData.shape)
    print("shape of reconData:", reconData.shape)
    print("\n\n\ndiff: ", diff)
    print("Reconstruct result is ", result)
    # for i in range(OrData.shape[0]):
    #     for j in range(OrData.shape[1]):
    #         if OrData[i][j] != reconData[i][j]:
    #             print("At (",i,", ",  j, "), the data is different. \nordata: ", OrData[i][j], "\nrecondata: ", reconData[i][j],)
    print("Reconstruct result is ", np.equal(OrData, reconData))

# pyplot.subplot(330 + 1)
# pyplot.imshow(element_2, cmap=pyplot.get_cmap('gray'))
# pyplot.show()
def runMyKnn():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X_2 = train_X.reshape(60000, 784)
    test_X_2 = test_X.reshape(10000, 784)

    print("test 7: ", len(test_X_2[8]))
    print("train 7: ", len(train_X_2[8]))
    k = 3
    wrong = np.array([])
    for i in range(len(test_X_2)):
        # print("test ", i, ": ", len(test_X_2[i]))
        # print("train: ", len(train_X_2[0]))
        if i % 500 == 0:
            print(i)
        if kNN(test_X_2[i], k, test_X_2, test_y) != test_y[i]:

            # print('For ', i, ', The predicted value is : ', kNN(test_X[i], k, test_X_2, test_y),
            #       ' and the true value is ',test_y[i])

            wrong = np.append(wrong, test_y[i])

    print(wrong)
    print(len(wrong))

    clas, freq = np.unique(wrong, return_counts=True)

    # Find the index of the maximum count
    most_frequent_index = np.argmax(freq)

    # Find the most frequent element
    most_frequent_element = clas[most_frequent_index]

    print("The hardest one for classification: ", most_frequent_element)

    print("Done for Knn")

def PCA_skl(X, X_test, n):
    pca = skl_PCA(n_components=n, svd_solver='full')
    pca.fit(X)
    # print(pca.explained_variance_ratio_)
    # print(len(pca.singular_values_))
    Z_pca = pca.fit_transform(X)
    index = pca.components_
    test_pca = pca.transform(X_test)
    return Z_pca, test_pca, pca.mean_

def MyPCA(df, n_com = -1):
    # checking shape
    print('Original Dataframe shape :', df.shape)

    # Mean
    X_mean = np.mean(df, axis=0)
    print("X_mean size is : ", X_mean.shape)
    print("X_mean is : ", X_mean)

    # Standard deviation
    X_std = np.std(df, axis=0)
    print("X_std size is : ", X_std.shape)

    # Get the index where the std is 0
    # std = 0 means the data is the same or constant
    constant_features = np.where(X_std == 0)[0]

    # Remove constant features from df, X_mean, and X_std
    newDf = np.delete(df, constant_features, axis=1)
    X_mean_g = np.delete(X_mean, constant_features)
    X_std = np.delete(X_std, constant_features)
    print('No 0 std shape :', newDf.shape)

    # Standardization
    Z = (newDf - X_mean_g) / X_std
    print("Z size is : ", Z.shape)

    # covariance
    c = np.cov(Z, rowvar=False)
    print("c shape: ", c.shape)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(c)
    # print('Eigen values:\n', eigenvalues)
    print('Eigen values Shape:', eigenvalues.shape)
    print('Eigen Vector Shape:', eigenvectors.shape)

    # Index the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    print("index: \n", idx)

    # # Sort the eigenvalues in descending order
    # eigenvalues = eigenvalues[idx]
    # print('Sorted Eigen values Shape:', eigenvalues)
    #
    # # sort the corresponding eigenvectors accordingly
    # eigenvectors = eigenvectors[:, idx]
    # print('Sorted Eigen vectors Shape:', eigenvectors)


    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    # print(explained_var)

    # How many component we want
    if n_com > 0:
        n_components = n_com
    else:
        n_components = np.argmax(explained_var >= 0.50) + 1
    print("Number of components: ", n_components)

    # PCA component or unit matrix
    # u = eigenvectors[:, :n_components]
    # print("u shape: ", u.shape)

    # Matrix multiplication
    # Z_pca = Z @ u
    Z_pca2 = Z @ pd.DataFrame(eigenvectors)

    Z_pca2 = Z_pca2.iloc[:, idx]

    Z_pca2 = Z_pca2.iloc[:60000, :n_components]

    eigenvectors = eigenvectors[:, idx]
    u = eigenvectors[:, :n_components]
    Z_pca1 = Z @ u

    # are_equal = np.array_equal(Z_pca1, Z_pca2)
    # print(are_equal)
    # Print the Pricipal Component values
    # print(Z_pca)
    print("Done for PCA Test")
    print("Start reconstruct!!")


    return Z_pca1, idx, X_mean

def ppt_PCA(df, n_com = -1):
    # checking shape
    print('Original Dataframe shape :', df.shape)

    # Mean
    X_mean = np.mean(df, axis=0)
    print("X_mean size is : ", X_mean.shape)
    print("X_mean is : ", X_mean)

    # Standard deviation
    X_std = np.std(df, axis=0)
    print("X_std size is : ", X_std.shape)

    # # Get the index where the std is 0
    # # std = 0 means the data is the same or constant
    # constant_features = np.where(X_std == 0)[0]
    #
    # # Remove constant features from df, X_mean, and X_std
    # df = np.delete(df, constant_features, axis=1)
    # X_mean_g = np.delete(X_mean, constant_features)
    # X_std = np.delete(X_std, constant_features)
    # print('No 0 std shape :', df.shape)

    # Standardization
    # Z = (df - X_mean) / X_std
    Z = df - X_mean
    print("Z size is : ", Z.shape)

    # covariance
    c = np.cov(df, rowvar=False)
    print("c shape: ", c.shape)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(c)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    print('Eigen values:\n', eigenvalues)
    print('Eigen vectors:\n', eigenvectors)
    print('Eigen values Shape:', eigenvalues.shape)
    print('Eigen Vector Shape:', eigenvectors.shape)

    # Index the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    print("index: \n", idx)

    # # Sort the eigenvalues in descending order
    # eigenvalues = eigenvalues[idx]
    # print('Sorted Eigen values Shape:', eigenvalues)
    #
    # # sort the corresponding eigenvectors accordingly
    # eigenvectors = eigenvectors[:, idx]
    # print('Sorted Eigen vectors Shape:', eigenvectors)


    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    # print(explained_var)

    # How many component we want
    if n_com > 0:
        n_components = n_com
    else:
        n_components = np.argmax(explained_var >= 0.50) + 1
    print("Number of components: ", n_components)

    # PCA component or unit matrix
    # u = eigenvectors[:, :n_components]
    # print("u shape: ", u.shape)

    # Matrix multiplication
    Z_pca2 = Z @ pd.DataFrame(eigenvectors)

    Z_pca3 = Z_pca2.iloc[:, idx]

    Z_pca_f = Z_pca3.iloc[:df.shape[0], :n_components]
    #
    # eigenvectors = eigenvectors[:, idx]
    # u = eigenvectors[:, :n_components]
    # Z_pca1 = Z @ u

    # are_equal = np.array_equal(Z_pca1, Z_pca2)
    # print(are_equal)
    # Print the Pricipal Component values
    # print(Z_pca)
    print("Done for PCA Test")

    print("Start reconstruct")

    recon_ppt(df, Z_pca2, eigenvectors, X_mean)
    return Z_pca_f, idx, X_mean

def PCA_TestData(data, index, n):
    if np.ndim(data) == 2:
        data = data[:, index]
        data = data[:, :n]
    elif np.ndim(data) == 1:
        data = data[index]
        data = data[:n]
    else:
        print("Wrong data shape, return original data. ")
        return data
    print("Test data shape: ", data.shape)
    return data
# Press the green button in the gutter to run the script.


def ppt_test():
    data = np.array([[2.5, 2.4], [.5, .7], [2.2,2.9], [1.9,2.2], [3.1,3.0], [2.3,2.7], [2.0,1.6], [1.0,1.1], [1.5,1.6], [1.1,.9]])
    data, index, mean = ppt_PCA(data, 2)
    print(data)
    print("Done ppt test")

if __name__ == '__main__':
    # ppt_test()

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X_2 = train_X.reshape(60000, 784)
    test_X_2 = test_X.reshape(10000, 784)

    numOfcom = 784
    k = 7

    # data_skl = PCA_skl(train_X_2, numOfcom)
    data_mine, index, myMean = ppt_PCA(train_X_2, numOfcom)
    # result = np.array_equal(data_mine, data_skl)

    print("Mine: ", data_mine[0])
    # print("skl: ", data_skl[0])

    pca_test_x = PCA_TestData(test_X_2, index, numOfcom)

    # # # Original data and knn
    # print("kNN with original data: ")
    # k_NN(train_X_2, test_X_2, train_y, test_y, k)
    #
    #
    # # skl pca data and knn
    # print("kNN with skl pca data: ")
    # x_skl, testx_skl, skl_Mean = PCA_skl(train_X_2, test_X_2, numOfcom)
    # k_NN(x_skl, testx_skl, train_y, test_y, k)

    # # My pca data and knn
    print("kNN with my pca data: ")
    k_NN(data_mine, pca_test_x, train_y, test_y, k)
    #
    # # PPT pca data and knn
    print("kNN with ppt pca data: ")
    ppt_X, ppt_index, ppt_mean = ppt_PCA(train_X_2, numOfcom)
    ppt_teat_x = PCA_TestData(test_X_2, ppt_index, numOfcom)
    k_NN(ppt_X, ppt_teat_x, train_y, test_y, k)


    print("done in main")
    # print("Mine and skl: ", result)











