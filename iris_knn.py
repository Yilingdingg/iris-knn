import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns

data = pd.read_csv('Data Sets\iris.csv')
print(data.info())
print(data.head())

x = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.4, random_state = 50)

# StandardScaler for the x, and le for the y

x_train = StandardScaler().fit_transform(x_train)
y_train = LabelEncoder().fit_transform(y_train)
x_test = StandardScaler().transform(x_test)
y_test = LabelEncoder().transform(y_test)

model = KNeighborsClassifier(n_neighbors = 5)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

matrix = confusion_matrix(y_test, y_predict)
sns.heatmap(matrix, annot = True, fmt = 'd')

plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

