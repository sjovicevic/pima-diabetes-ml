import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
import numpy as np

df = pd.read_csv("../data/pima-data.csv")

# Deleted skin column because of redundant data
del df['skin']

# Changing diabetes values from True/False to 1/0 - for helping algorithm
diabetes_map = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

# Calculating the number of True/False cases
num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])

# Calculating the percentage of True/False cases
percentage_of_num_true = num_true / (num_true + num_false) * 100
percentage_of_num_false = num_false / (num_true + num_false) * 100

# Outputting the True/False cases
print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, percentage_of_num_true))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, percentage_of_num_false))


# 70% for training, 30% for testing

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))

fill_0 = SimpleImputer(missing_values = 0, strategy="mean")

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())

print(nb_model.get_params())
