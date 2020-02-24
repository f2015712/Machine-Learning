import matplotlib.pyplot as plt 
from matplotlib import style 
style.use('ggplot')
import numpy as np 
from sklearn.cluster import KMeans

X=np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])
#plt.scatter(X[:,0],X[:,1],s=150)
#plt.show()
#parameter is the number of clusters
clf=KMeans(n_clusters=4)
clf.fit(X)
#get the centroids after clustering the data points
centroids=clf.cluster_centers_
#get the labels i.e for 2 clusters, labels will be 0 or 1
labels=clf.labels_
colors=["g.","r.","c.","b.","k.","o."]

for i in range(len(X)):
	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
plt.scatter(centroids[:,0],centroids[:,1],marker='x', s=150,linewidth=5)
plt.show()

