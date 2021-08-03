from scipy.sparse.construct import rand
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from matplotlib import pyplot as plt

# Fetch from openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# print(X.shape, y.shape)
# print(np.min(X), np.max(X))
# print(y[0:5])

# segment out only data of 0-3 digits
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6969)

print(X_test.shape, y_test.shape)
# print(np.min(X), np.max(X))
# print(y[0:5])

mlp=MLPClassifier(
  hidden_layer_sizes=(50, 50), 
  max_iter=200, alpha=1e-4,
  solver='sgd', random_state=2)

mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))

# This is supposed to show and validate 10 first predictions
# for j in range(10):    
#     y_pred = mlp.predict(X_test)
#     incorrect = X_test[y_pred != y_test]
#     plt.matshow(incorrect[j].reshape(28, 28), cmap=plt.cm.gray)
#     plt.xticks(())
#     plt.yticks(())

#     print(f"True value: {y_test[j]}")
#     print(f"Predicted : {mlp.predict(X_test[j])}")

#     plt.show()

# # Visualise what the hidden nodes are doing
# fig, axes = plt.subplots(2, 3, figsize=(5, 4))
# for i, ax in enumerate(axes.ravel()):
#     coef = mlp.coefs_[0][:, i]
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.set_title(i + 1)
# plt.show()