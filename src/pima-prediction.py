import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../data/pima-data.csv")

del df['skin']
diabetes_map = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])

percentage_of_num_true = num_true / (num_true + num_false) * 100
percentage_of_num_false = num_false / (num_true + num_false) * 100

print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, percentage_of_num_true))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, percentage_of_num_false))



