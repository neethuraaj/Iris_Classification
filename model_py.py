# -*- coding: utf-8 -*-
"""model.py

'''
Created on 01-April-2023

@author: Neethu Raj
'''

# PythonLibraries
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

"""# import data"""

filename="iris.xls"

data=pd.read_excel(filename)

"""# Split the dataset into the training and test sets"""

# -----------------Split Dependent and Independent variable 

#Target Variable\Dependent variable Y = data.Classification
# Independent variable x contains all the input variables such as independent features


x=data.iloc[:,:4]
y=data.iloc[:,4]

# ----------------- Split the data into Test & Train data ------------

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


"""# KNN model"""

#  Higher accuracy for k values 3,7 & 9
knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='euclidean')
knn.fit(x_train, y_train)

#Saving the model to disk
pickle.dump(knn,open('model.pkl','wb') )