import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report

# creating dataset
dataset = pd.read_csv("breast-cancer.csv")

# label encoding
labelencoder = LabelEncoder()
dataset["diagnosis"] = labelencoder.fit_transform(dataset["diagnosis"].values)

X = dataset.drop("diagnosis", axis=1)
y = dataset.loc[:, "diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# # Bernoulli Naive Bayes
# model=BernoulliNB()
# model.fit(X_train, y_train)

# # Multinominal Naive Bayes
# model=MultinomialNB()
# model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(classification_report(y_test, prediction))
