from numpy import mod
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Acquire data and store it in a dataFrame
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
# Convert values to binary (or rather Booleans)
df['male'] = df['Sex'] == 'male'
# Assign arrays derived from particular columns of our dataFrame to X and Y respectively 
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

# Using LogisticRegression to compute passenger's chances of survival on our data
model = LogisticRegression()
model.fit(X, y)

# Predict what happens to first n passenger's in our data based on our model
n = 10
print(model.predict(X[:n]))
# The actual state of the n first passenger's from our data set
print(y[:n])

# Array of Predicted outcomes 
yPredicted = model.predict(X)
# Compute how many answers we've gotten correctly
correctPredictions = (y == yPredicted).sum()
# Percentage of how many outcomes we predicted correctly
correctPercentage = correctPredictions / y.shape[0]
print(f"{correctPercentage * 100} %")
# Or the built in method
score = model.score(X, y)
print(f"{score * 100} %")


# Print values of our line computed by fitting our data to a LogisticRegression model
# print(model.coef_, model.intercept_)

# Print arrays derived from our dataFrame
# print(X)
# print(y)
