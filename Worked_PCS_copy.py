# 2023.9.26
# Implementation of kNN for Mnist data collection
# 60k training data, and 10k testing data
# Find the digit with the biggest difficulty classify

from keras.datasets import mnist
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

def recon_ppt(OrData, transfData, eigvect, mean, zero_idx):
    try:
        # transpEigvec = np.linalg.pinv(eigvect)
        transpEigvec, residuals, r, s = np.linalg.lstsq(eigvect)
        print("Matrix is invertible.")
    except np.linalg.LinAlgError:
        print("Matrix is not invertible.")
        return 0
    print("shape of transfData:", transfData.shape)
    print("shape of transpEigvec:", transpEigvec.shape)

    # transpEigvec = transpEigvec[:transfData.shape[1], :]
    # AdData = transfData @ np.transpose(transpEigvec)
    AdData = transfData @ transpEigvec
    print("shape of AdData:", AdData.shape)
    reconData = AdData + mean
    result = np.allclose(OrData, reconData) | np.array_equal(OrData, reconData)
    diff = OrData - reconData
    print("shape of OrData:", OrData.shape)
    print("shape of reconData:", reconData.shape)
    print("\n\n\ndiff: ", diff)
    print("Reconstruct result is ", result)

    # np.savetxt('diff.txt', diff, comments='')

    non_zero_columns = np.where(np.any(diff != 0, axis=0))[0]
    # non_zero_columns = non_zero_columns.reshape(1, -1)
    # new_arr = np.concatenate((non_zero_columns, zero_idx))
    #
    # new_arr.sort()
    # print("Shape of the new arr: ", new_arr.shape)
    #
    # # Check if each element is one greater than the previous element
    # for i in range(len(new_arr)):
    #     if i != new_arr[i]:
    #         print("Doesn't include all indices")

    print("False indices: ", str(non_zero_columns))
    print("Zero indices: ", str(zero_idx))

def PCA_skl(X, X_test, n):
    pca = skl_PCA(n_components=n, svd_solver='full')
    pca.fit(X)
    # print(pca.explained_variance_ratio_)
    # print(len(pca.singular_values_))
    Z_pca = pca.fit_transform(X)
    test_pca = pca.transform(X_test)
    return Z_pca, test_pca

def redo1_TestData(Test_Data, eigvec):
    test_mean = np.mean(Test_Data, axis = 0)
    stand_test = Test_Data - test_mean
    new_Test = stand_test @ eigvec
    return new_Test

def redo1_PCA(df, n_com=-1):

    # get the mean and the std
    X_mean = np.mean(df, axis = 0)
    X_std = np.std(df, axis = 0)

    both_zero = np.where((X_mean == 0) & (X_std == 0))
    std_zero = np.where(X_std == 0)

    # newDf = np.delete(df, both_zero, axis=1)
    # X_mean_g = np.delete(X_mean, both_zero)
    # X_std_g = np.delete(X_std, both_zero)
    # print('No 0 std shape :', newDf.shape)
    #
    # Z = (newDf - X_mean_g) / X_std_g

    Z = (df - X_mean)
    print("Z size is : ", Z.shape)

    c = np.cov(Z, rowvar=False)
    print("c shape: ", c.shape)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(c)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    # print('Eigen values:\n', eigenvalues)
    # print('Eigen vectors:\n', eigenvectors)
    print('Eigen values Shape:', eigenvalues.shape)
    print('Eigen Vector Shape:', eigenvectors.shape)

    # Index the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    # print("index: \n", idx)

    # Sort the eigenvalues in descending order
    eigenvalues = eigenvalues[idx]

    # sort the corresponding eigenvectors accordingly
    eigenvectors = eigenvectors[:, idx]

    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    # print(explained_var)

    # How many component we want
    if n_com > 0:
        n_components = n_com
    else:
        n_components = np.argmax(explained_var >= 0.50) + 1
    print("Number of components: ", n_components)

    u = eigenvectors[:, :n_components]
    print("u shape: ", u.shape)

    Z_pca = Z @ pd.DataFrame(u)

    return Z_pca, both_zero, u, X_mean


if __name__ == '__main__':
    # ppt_test()

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X_2 = train_X.reshape(60000, 784)
    test_X_2 = test_X.reshape(10000, 784)

    numOfcom = 157
    k = 7

    redo_trainX, zero_idx, eigvec, X_mean = redo1_PCA(train_X_2, numOfcom)

    redo_testX = redo1_TestData(test_X_2, eigvec, )
    recon_ppt(train_X_2, redo_trainX, eigvec, X_mean, zero_idx)

    # # # Original data and knn
    # print("kNN with original data: ")
    # k_NN(train_X_2, test_X_2, train_y, test_y, k)
    #
    #
    # # skl pca data and knn
    # print("kNN with skl pca data: ")
    # x_skl, testx_skl = PCA_skl(train_X_2, test_X_2, numOfcom)
    # k_NN(x_skl, testx_skl, train_y, test_y, k)
    #
    # print("Redo PCA: ")
    # k_NN(redo_trainX, redo_testX, train_y, test_y, k)


    print("done in main")
    # print("Mine and skl: ", result)











