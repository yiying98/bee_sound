import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.decomposition import TruncatedSVD 

iris = datasets.load_iris() 
X = iris.data 
y = iris.target  # Labels

# Visualize result using SVD
svd = TruncatedSVD(n_components=2) 
X_reduced = svd.fit_transform(X)


# Initialize scatter plot with x and y axis values
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=25,label=["normal", "queenless", "virus"])
plt.legend()
plt.savefig("iris_plot.png")
plt.show()
