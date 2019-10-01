# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
# Always have X as a matrix and y as a vector
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values



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
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2 ,random_state=0)
"""




"""
# fitiing the SLR model in the matrix created 
linreg_2 = LinearRegression()
linreg_2 = linreg_2.fit(X_pol, y)
y_pred_pol = linreg_2.predict(X_pol)
"""
"""
# Better graph 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
"""


# fitting the model ("regressor" is created here)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)





# predicting the output for a particular value ([[]]) is to make sure that the inout for the transform function is a array
# predict and transform operations are used to make sure that the proper scaling is maintained as we are involving feature scaling in the process
y_pred = regressor.predict(np.array([[6.5]]))

# plotting the polynomial linear regression model
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X),color='blue')
plt.show()

# graph of better resolution
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')
plt.show()











