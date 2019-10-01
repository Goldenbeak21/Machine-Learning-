# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,1].values

"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
#imputer = imputer.fit(X[:,1:3])
#X[:,1:3] = imputer.transform(X[:,1:3])
X[:,1:3] = imputer.fit_transform(X[:,1:3])
"""

"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_X = labelencoder_X.fit(X[:,0])
X[:,0] = labelencoder_X.transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
labelencoder_y = labelencoder_y.fit(y)
y = labelencoder_y.transform(y)
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

"""

FEATURE SCALING IS DONE BY THE LIBRARY ITSELF FOR SIMPLE LINEAR REGRESSION

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X = sc_X.fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)
"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.show()


plt.scatter(X_test,y_test,color='red')
plt.plot(X_test, y_pred,color='blue')
plt.show()

