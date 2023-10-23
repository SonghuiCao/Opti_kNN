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

# Distance calculation for my own kNN
def distance(x, y):
    # flatten() makes 2d array into 1d array, then do the distance calculation
    return np.linalg.norm(np.array(x.flatten()) - np.array(y.flatten()))

# My own kNN function
def My_kNN(x, k, data, label):

    distances =[distance(x, data[i]) for i in range(len(data))]
    # Use "np.argpartition". It does not sort the entire array.
    # It only guarantees that the kth element is in sorted position
    # and all smaller elements will be moved before it.
    # Thus the first k elements will be the k-smallest elements.
    idx = np.argpartition(distances, k)

    clas, freq = np.unique(label[idx[:k]], return_counts=True)

    return clas[np.argmax(freq)]

# Sklearn kNN function
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

    # Get the accuracy for the classifier
    knn_accuracy = accuracy_score(y_test, knn_pred)

    # Print the result
    print("k-NN Accuracy:", knn_accuracy)
    print(f"k-NN Training Time: {knn_train_time:.5f} seconds")

# Reconstruct function for ppt
# reconData = eigvec^-1 @ transfData(transformed data) + mean
def recon_ppt(OrData, transfData, eigvect, mean):
    # Get the reconstruct data
    transpEigvec = np.transpose(eigvect)
    AdData = transfData @ transpEigvec
    reconData = AdData + mean

    # Compare with the original data
    result = np.allclose(OrData, reconData) | np.array_equal(OrData, reconData)
    print("Reconstruct result is ", result)

    # -------- Try to find where is the problems --------
    # From the experiment, the most error case is happened when original data is 0 but reconstruct is not 0
    # diff = OrData - reconData
    # print("\n\n\ndiff: ", diff)
    # for i in range(OrData.shape[0]):
    #     for j in range(OrData.shape[1]):
    #         if OrData[i][j] != reconData[i][j]:
    #             print("At (",i,", ",  j, "), the data is different. \nordata: ", OrData[i][j], "\nrecondata: ", reconData[i][j],)
    # print("Reconstruct result is ", np.equal(OrData, reconData))

# PCA function using sklearn
def PCA_skl(X_train, X_test, n):
    pca = skl_PCA(n_components=n, svd_solver='full')
    pca.fit(X_train)

    Z_pca = pca.fit_transform(X_train)
    # print(pca.components_)
    # print(pca.components_.shape)
    cov = pca.get_covariance()
    print("cov from sklearn: ", cov)
    test_pca = pca.transform(X_test)
    return Z_pca, test_pca, pca.mean_, cov

# PCA from Powerpoint
def ppt_PCA(df, n_com = -1):

    # Mean
    X_mean = np.mean(df, axis=0)
    # print("X_mean size is : ", X_mean.shape)
    # print("X_mean is : ", X_mean)

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
    Z = (newDf - X_mean_g) / X_std         # Another way to standardization
    # Z = df - X_mean
    print("Z size is : ", Z.shape)

    # covariance
    c = np.cov(Z, rowvar=False)
    print("c shape: ", c.shape)
    print("cov from my pca: ", c)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(c)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    print('Eigen values:\n', eigenvalues)
    # print('Eigen vectors:\n', eigenvectors)
    print('Eigen values Shape:', eigenvalues.shape)
    print('Eigen Vector Shape:', eigenvectors.shape)

    # Index the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    print("index: \n", idx)
    print("index shape: \n", idx.shape)


    eigenvectors = eigenvectors[:, idx]
    pc1 = eigenvectors[:, 0]
    print("pc1: ", pc1)
    feaIdx = pc1.argsort()
    print("feaind: ", feaIdx)

    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    # print("explained var: ", explained_var)

    # How many component we want
    if n_com > 0:
        n_components = n_com
    else:
        n_components = np.argmax(explained_var >= 0.95) + 1
    print("Number of components: ", n_components)

    # Matrix multiplication
    Z_pca2 = Z @ pd.DataFrame(eigenvectors)                 # original order and eigvec dot product
    Z_pca3 = Z_pca2.iloc[:, idx]                            # rearrange Z_pca with index order from sorting eigvalues
    Z_pca_f = Z_pca3.iloc[:df.shape[0], :n_components]      # Take top n_components Z_pca

    print("Done for PCA Test")

    print("Start reconstruct")
    recon_ppt(df, Z_pca2, eigenvectors, X_mean)

    return Z_pca_f, feaIdx, X_mean, c

# Reduce the dimensions of test X
def PCA_TestData(data, index, n):
    if np.ndim(data) == 2:
        # rearrange test_x
        data = data[:, index]
        # take top n_components
        data = data[:, :n]
    elif np.ndim(data) == 1:
        data = data[index]
        data = data[:n]
    else:
        print("Wrong data shape, return original data. ")
        return data
    print("Test data shape: ", data.shape)
    return data

# Example from Powerpoint for check purpose
def ppt_test():
    data = np.array([[2.5, 2.4], [.5, .7], [2.2,2.9], [1.9,2.2], [3.1,3.0], [2.3,2.7], [2.0,1.6], [1.0,1.1], [1.5,1.6], [1.1,.9]])
    data, index, mean = ppt_PCA(data, 2)
    print(data)
    print("Done ppt test")

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X_2 = train_X.reshape(60000, 784)
    test_X_2 = test_X.reshape(10000, 784)

    # parameters for PCA and kNN
    numOfcom = 716
    k = 7

    # Do the pca for train_X and get test_X
    train_x_myPCA, index, myMean, myCov = ppt_PCA(train_X_2, numOfcom)
    pca_test_x = PCA_TestData(test_X_2, index, numOfcom)

    x_skl, testx_skl, skl_Mean, skl_cov = PCA_skl(train_X_2, test_X_2, numOfcom)
    # # Original data and knn
    # print("kNN with original data: ")
    # k_NN(train_X_2, test_X_2, train_y, test_y, k)
    #
    #
    # # skl pca data and knn
    # print("kNN with skl pca data: ")
    # x_skl, testx_skl, skl_Mean = PCA_skl(train_X_2, test_X_2, numOfcom)
    # k_NN(x_skl, testx_skl, train_y, test_y, k)
    #
    # My pca data and knn
    print("kNN with my pca data: ")
    k_NN(train_x_myPCA, pca_test_x, train_y, test_y, k)

    print("done in main")











