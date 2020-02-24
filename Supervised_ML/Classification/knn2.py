#KNN can be used for non-linear data
import numpy as np 
from math import sqrt 
import matplotlib.pyplot as plt 
import warnings 
from matplotlib import style 
from collections import Counter
import pandas as pd 
import random 
style.use('fivethirtyeight')

dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features=[5,7]
#s=100 denotes the size of the data point 

"""
for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1], s=100, color=i)
plt.scatter(new_features[0],new_features[1],s=100)
plt.show()
"""
#here, we calculate the top 3 distances since k=3
def k_nearest_neighbors(data,predict,k=3):
	if len(data)>=k:
		warnings.warn('K is set to a value less than total groups')
	distances=[]
	#data is a dictionary
	for group in data:
		for features in data[group]:
			#np.linalg calculates the normalized distance i.e the euclidean distance between 2 arrays
			euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
			#store the class and distance of that class from the required point
			distances.append([euclidean_distance,group])
	#sort the distances in ascending order and get the top k values
	distances.sort() 
	#get top-k entries only
	distances=distances[:k]
	#store only the 1st index entries since that contains the labels
	votes=[distances[i][1] for i in range(len(distances)) ]

	#get the most common entry
	vote_result=Counter(votes).most_common(1)[0][0]
	return vote_result

"""

result=k_nearest_neighbors(dataset,new_features,k=3)
print(result)

"""

df=pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
#every row gets into full_data as an array, i.e full_data is an array of arrays
full_data=df.astype(float).values.tolist()
random.shuffle(full_data)
#defining the test_set_size
test_size=0.2
#dictionary containing labels as the keys for training set
train_set={2:[],4:[]}
#dictionary containing labels as the keys for the test set
test_set={2:[],4:[]}

#get the bottom 20% data
train_data=full_data[:-int(test_size*len(full_data))]
#get the top 20% data
test_data=full_data[-int(test_size*len(full_data)):]

#the last index contains the label
for i in train_data:
	train_set[i[-1]].append(i[:-1])
for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct=0
total=0

for group in test_set:

	for data in test_set[group]:
		vote=k_nearest_neighbors(train_set,data,k=5)
		if group==vote:
			correct+=1
		total+=1
print('Accuracy: ',correct/total)
