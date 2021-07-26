from numpy import mod
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split 

# sensiticity calculation (same as recall)
sensitivity_score = recall_score
# specificity calculation, 2nd array from fscore support contains [1] - positive class recall - sensitivity, [0] - negative class recall - specificity
def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]

# Acquire data and store it in a dataFrame
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
# Convert values to binary (or rather Booleans)
df['male'] = df['Sex'] == 'male'
# Assign arrays derived from particular columns of our dataFrame to X and Y respectively 
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

# sklearn's built-in method to split data randomly into 60% of training data and 40% test data (default ratio 75/25), we can specify a seed (f.e. 1221)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1221)

print("whole dataset:", X.shape, y.shape)
print("training set:", X_train.shape, y_train.shape)
print("test set:", X_test.shape, y_test.shape)

# Using LogisticRegression to compute passenger's chances of survival on our data
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict what happens to first n passenger's in our data based on our model
# n = 10
# print(model.predict(X[:n]))

# The actual state of the n first passenger's from our data set
# print(y[:n])

# Array of Predicted outcomes, here 
# yPredicted = model.predict(X_test)
# Array of Predicted outcomes, but with a implicit treshold = 0.75
yPredicted = model.predict_proba(X_test)[:, 1] > 0.75
# # Compute how many answers we've gotten correctly
# correctPredictions = (y == yPredicted).sum()
# # Percentage of how many outcomes we predicted correctly
# correctPercentage = correctPredictions / y.shape[0]
# print(f"{correctPercentage * 100} %")

# Or just use the sklearn's built in method
score = model.score(X_test, y_test)
# print(f"{score * 100} %")

# Print values of our line computed by fitting our data to a LogisticRegression model
# print(model.coef_, model.intercept_)

# sklearn's built-in model describing metrics
print("accuracy:", accuracy_score(y_test, yPredicted))
print("precision:", precision_score(y_test, yPredicted))
print("recall:", recall_score(y_test, yPredicted))
print("f1 score:", f1_score(y_test, yPredicted))

# sklearn's built-in confusion matrix (shows negatives first! 'Transposed matrix' because positive - 1 and negative - 0)
print(confusion_matrix(y_test, yPredicted))

print("sensitivity:", sensitivity_score(y_test, yPredicted))
print("specificity:", specificity_score(y_test, yPredicted))
