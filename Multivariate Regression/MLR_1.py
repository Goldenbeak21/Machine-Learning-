# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train_data.csv')
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,4].values

"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
#imputer = imputer.fit(X[:,1:3])
#X[:,1:3] = imputer.transform(X[:,1:3])
X[:,1:3] = imputer.fit_transform(X[:,1:3])
"""


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
labelencoder_X = labelencoder_X.fit(X[:,3])
X[:,3] = labelencoder_X.transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
"""
labelencoder_y = LabelEncoder()
labelencoder_y = labelencoder_y.fit(y)
y = labelencoder_y.transform(y)
"""
"""
# MAKING SURE WE WON'T CONSIDER A DEPENDENT VARIABLE IN OUR EQUATION
X = X[:,1:]
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

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
regressor.fit(X, y)
X_test1 = pd.read_csv('test_data.csv') 
y_pred = regressor.predict(X_test1)

#WE ADD A COLUMN OF ONES AS THE BACKWARD ELIMINATION LIBRARY DOES NOT TAKE CARE OF IT
"""

MANUAL BACKWARD ELIMINATION BY COMPARING THE P VALUES

import statsmodels.formula.api as sm
X = np.append(np.ones((2000,1)).astype(int), X, axis = 1)
X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS( y , X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3]]
regressor_OLS = sm.OLS( y , X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS( y , X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[3,4,5]]
regressor_OLS = sm.OLS( y , X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[3,4]]
regressor_OLS = sm.OLS( y , X_opt).fit()
regressor_OLS.summary()
"""
"""

OWN CODE FOR AUTOMIATIC BACKWARD ELIMINATION

import statsmodels.formula.api as sm
def backwardelimination (X_f , sl):
    numvar = len(X_f[0])
    for i in range(0,numvar):
            regressor_OLS = sm.OLS( y , X_f).fit()
            maxp = max(regressor_OLS.pvalues).astype(float)
            if maxp > sl:
                for j in range(0, numvar-i):
                    if regressor_OLS.pvalues[j].astype(float)==maxp :
                        X_f = np.delete(X_f , j , 1 )
    regressor_OLS.summary()
    return X_f
    
    
sl=0.05
X_p = X[:, [0, 1, 2, 3, 4, 5]]
X_model = backwardelimination(X_p,sl)
        
"""
"""

COPIED CODE FROM THE COURSE

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

"""





























