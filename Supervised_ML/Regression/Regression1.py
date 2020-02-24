import pandas as pd 
import quandl
import math
import numpy as np 
#preprocessing is used for feature scaling i.e put the features between +1 and -1
#cross-validation is used for splitting data into training and testing data sets

from sklearn import preprocessing,svm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import datetime
import matplotlib.pyplot as plt # using graph
from matplotlib import style # using styles
style.use('ggplot') #specify the type of plot

#loading data into the dataframe
df=quandl.get('WIKI/GOOGL')
#df=quandl.get('BATS/EDGA_XLTY')
#print(df.head())
#get top few entries of the data frame
#print(df.head())
#get the dataframe with the required columns
#selecting important features
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#print(df.head())
#creating new columns
df['PCT_Change']=((df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'])*100.0
#print(df.head())
forecast_col='Adj. Close'
#filling NaN values
df['label']=df[forecast_col]
#df.fillna(-99999,inplace=True)
df.dropna(inplace=True) 
# Drop entries which contains NaN values
#print(df.head())
#X contains the input features. We have all the features except label column
X=np.array(df.drop(['label'],1))
y=np.array(df['label'])
#scaling done on features
X=preprocessing.scale(X)
#print(len(X),len(y))
#split 20% randomly shuffled data into test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#clf=svm.SVR()
#by default, the curve is linear. To change it, we need to modify the kernel
#clf=svm.SVR(kernal='poly')
clf=LinearRegression()
#clf=LinearRegression(n_jobs=10) - means run 10 jobs
clf.fit(X_train,y_train)

#print(accuracy)
#clf.predict()	



#using pickle
#'wb'-write
#'rb' -read
# open the linearregression pickle and dump the classifier i.e clf
#linearregression is the name with which this pickle file gets saved
import pickle

with open('linearregression.pickle','wb') as f:
	pickle.dump(clf,f)
#now we open the pickle for reading data 
pickle_in=open('linearregression.pickle','rb')

clf=pickle.load(pickle_in)


accuracy=clf.score(X_test,y_test)
print(accuracy)