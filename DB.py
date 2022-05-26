from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


plt.figure(figsize=(8, 8))


# %% function
def draw_db(trainx, trainy, models, x_scale=[-10, 10], y_scale=[-10, 10], colors=['g', 'b', 'r', 'y']):
    plt.figure(figsize=(8, 8))
    xx, yy = np.meshgrid(np.linspace(x_scale[0], x_scale[1], 1000), np.linspace(y_scale[0], y_scale[1], 1000))
    z = np.c_[xx.ravel(), yy.ravel()]
    for i in np.unique(trainy):
        y_index = np.argwhere(trainy == i)
        data_x = trainx[y_index,[0,1]]
        plt.scatter(data_x[:, 0], data_x[:, 1], label='0')

    for n,model in enumerate(models):
        model.fit(trainx, trainy)
        z_lg = model.predict(z).reshape(xx.shape)
        plt.contour(xx, yy, z_lg, colors=colors[n])

    plt.legend()
    plt.show()


# %% test
if __name__ == '__main__':
    trainx = np.array([[1.5, 0], [2.7, 4], [1.7, 3], [3.5, 2], [1.9, 6], [4, 5], [2.5, 2], [0, 3]])
    trainy = np.array([0,1,1,0,0,0,1,0])
    plt.figure(figsize=(8, 8))
    knn = KNeighborsClassifier(n_neighbors=1)
    lsvm1 = SVC(kernel='linear',C=1000)
    lg = LogisticRegression()
    gnb = GaussianNB()
    dt = DecisionTreeClassifier()


    draw_db(trainx, trainy, models=[knn, lg, gnb, dt], x_scale=[0, 7], y_scale=[-1, 6])

