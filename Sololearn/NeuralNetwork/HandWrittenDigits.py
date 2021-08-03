from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# load the dataset
X, y = load_digits(return_X_y=True)
# split it 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
# create an MLP
mlp = MLPClassifier()
# train it
mlp.fit(X_train, y_train)

# show the first datapoint
x = X_test[1]
# plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
# plt.xticks(())  # remove x tick marks
# plt.yticks(())  # remove y tick marks
# plt.show()

# and try to predict it
print(mlp.predict([x]))

# score the model
print(mlp.score(X_test, y_test))

# Mask incorrectly guessed images
y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test] 

# Show the first of them
for j in range(len(incorrect)):    
    plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    print("true value:", incorrect_true[j])
    print("predicted value:", incorrect_pred[j])