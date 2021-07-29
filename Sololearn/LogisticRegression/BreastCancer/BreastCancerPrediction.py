import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# acquire data from breast cancer dataset
cancer_data = load_breast_cancer()

# print(cancer_data.keys())
# print(cancer_data['data'].shape)

# Create a data frame based on data from breast cancer dataset
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])

# print(cancer_data['target'].shape)
# print(cancer_data['target_names'])

# Add target data as a column to our data frame
df['target'] = cancer_data['target']

# print(df.head())

# Create ndArrays from specific data frame columns, X - data, Y - target data
X = df[cancer_data.feature_names].values
y = df['target'].values

# Build the model and fit data
model = LogisticRegression(solver='liblinear')
model.fit(X, y)

# print(f"1st datapoint: {model.predict([X[0]])[0]} vs target: {cancer_data['target'][0]}")

# How well the model predicts 
print(model.score(X, ))