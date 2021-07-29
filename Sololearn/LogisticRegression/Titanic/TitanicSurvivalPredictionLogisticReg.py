from numpy import mod
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

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

# ROC curve 
y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])

# # Plot ROC
# plt.plot(fpr, tpr)
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('1 - specificity')
# plt.ylabel('sensitivity')
# plt.show()

# Create 2 logistic regression models. 1 with all features and 2 with only two of them.
# Calculate their corresponding area under their ROC curves (AUC) and print those values
model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred_proba1 = model1.predict_proba(X_test)
print("model 1 AUC score:", roc_auc_score(y_test, y_pred_proba1[:, 1]))

model2 = LogisticRegression()
model2.fit(X_train[:, 0:2], y_train)
y_pred_proba2 = model2.predict_proba(X_test[:, 0:2])
print("model 2 AUC score:", roc_auc_score(y_test, y_pred_proba2[:, 1]))

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
# print(confusion_matrix(y_test, yPredicted))

print("sensitivity:", sensitivity_score(y_test, yPredicted))
print("specificity:", specificity_score(y_test, yPredicted))

# We can use k-fold validation to make most of a limitingly small dataset
# Here we split and shuffle dataset into 5 training/test sets and calculate each output model's score
scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
# Here we print all the scores and one average score calculated from them
print("--K-fold validation--")
print(f"Scores: {scores}")
print(f"Mean of scores: {np.mean(scores)}")
