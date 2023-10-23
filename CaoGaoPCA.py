import pandas as pd
import numpy as np

# Here we are using inbuilt dataset of scikit learn
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA as skl_PCA


def pca_func(df, n):
    # checking shape
    print('Original Dataframe shape :',df.shape)

    # Input features
    X = df[cancer['feature_names']]
    print('Inputs Dataframe shape :', X.shape)
    print(cancer.feature_names)

    # Mean
    X_mean = X.mean()
    print("X_mean size is : ", X_mean.shape)

    # Standard deviation
    X_std = X.std()
    print("X_std size is : ", X_std.shape)

    # Standardization
    Z = (X - X_mean) / X_std

    print("Z size is : ", Z.shape)
    # covariance
    c = Z.cov()
    print("c shape: ", c.shape)
    # Plot the covariance matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.heatmap(c)
    # plt.show()

    eigenvalues, eigenvectors = np.linalg.eig(c)
    print('Eigen values:\n', eigenvalues)
    print('Eigen values Shape:', eigenvalues.shape)
    print('Eigen Vector Shape:', eigenvectors.shape)

    # Index the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    # print("index: \n", idx)

    # Sort the eigenvalues in descending order
    eigenvalues = eigenvalues[idx]

    # sort the corresponding eigenvectors accordingly
    eigenvectors = eigenvectors[:,idx]
    pc1 = eigenvectors[:, 0]
    print("pc1: ", pc1)
    feaIdx = pc1.argsort()
    print("feaind: ", feaIdx)


    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    # print(explained_var)

    n_components = np.argmax(explained_var >= 0.50) + 1
    print("Number of components: ", n)

    # PCA component or unit matrix
    u = eigenvectors[:,:n]
    print("u shape: ", u.shape)


    # # plotting heatmap
    # plt.figure(figsize =(7, 7))
    # sns.heatmap(pca_component)
    # plt.title('PCA Component')
    # plt.show()

    # Matrix multiplication or dot Product
    Z_pca = Z @ u
    # Rename the columns name
    # Print the Pricipal Component values
    print(Z_pca)
    return Z_pca


def PCA_skl(X, n):
    pca = skl_PCA(n_components=n, svd_solver='full')
    pca.fit(X)
    # print(pca.explained_variance_ratio_)
    print(pca.mean_)
    # print(len(pca.singular_values_))
    Z_pca = pca.fit_transform(X)
    index = pca.components_
    # test_pca = pca.transform(X_test)
    return Z_pca

if __name__ == '__main__':
    # instantiating
    cancer = load_breast_cancer(as_frame=True)
    # creating dataframe
    df = cancer.frame

    numOfcom = 17
    pca_x = pca_func(df, numOfcom)
    pca_skl_x = PCA_skl(df, numOfcom)
    print("done in main")
    print(np.array_equal(pca_x, pca_skl_x))
    print("mean for my pca: ", np.mean(pca_x, axis=0))
    print("mean for skl pca: ", np.mean(pca_skl_x, axis=0))



