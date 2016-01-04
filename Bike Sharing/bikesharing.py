import csv
import numpy as np
import math

trainPercent = 0.75

f = open('hour.csv')
l = list(csv.reader(f))

discreteColNames = ['season','mnth','hr','holiday','weekday','workingday','weathersit']
continousColNames = ['atemp']
titles = l[0]
l=np.asarray(l[1:])

import random
random.shuffle(l)


discreteColIndexes = []
continousColIndexes = []
for i in discreteColNames:
	discreteColIndexes.append(titles.index(i))
	
for i in continousColNames:
	continousColIndexes.append(titles.index(i))

# preprocessing discrete data as described here http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
from sklearn.preprocessing import OneHotEncoder

dX = l[:,discreteColIndexes].astype(int)

enc = OneHotEncoder()
a = enc.fit(dX)

x = l[:,continousColIndexes].astype(float)
y = l[:,-1].astype(int)


processedDX = enc.transform(dX).toarray()

# done preprocessing discrete data
tempX = []

for i in range(len(processedDX)):
	tempX.append(np.append(x[i],processedDX[i]))

x = tempX

len_train = int(len(x)*trainPercent)

x_train = x[:len_train]
y_train = y[:len_train]
x_test = x[len(x)-len_train:]
y_test = y[len(x)-len_train:]

print len_train,'training samples'
print len(x)-len_train,'testing samples'

from sklearn import tree,ensemble
regr = tree.DecisionTreeRegressor()
regr.fit(x_train,y_train)
acc = regr.score(x_test,y_test)
print 'Decision Trees Score',acc

regr = ensemble.RandomForestRegressor()
regr.fit(x_train,y_train)
acc = regr.score(x_test,y_test)

print 'Random Forest Score',acc
