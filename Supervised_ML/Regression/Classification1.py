from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('fivethirtyeight')

xs=np.array([1,2,3,4,5,6],dtype=np.float64)
ys=np.array([5,4,6,5,6,7],dtype=np.float64)

#plt.plot(xs,ys) - line plot
#plt.scatter(xs,ys) - scatter plot
#plt.show()

def squared_error(ys_original,ys_line):
	return sum((ys_line-ys_original)**2)

def best_fit_slope_and_intercept(xs,ys):
	m=((mean(xs)*mean(ys))-mean(xs*ys))/(((mean(xs))**2)-(mean(xs**2)))
	c=mean(ys)-m*(mean(xs))
	return m,c
#coefficient of deteremination describes how well the best fit line is
def coefficient_of_determination(ys,ps):
	n=len(xs)
	#r=(n*(sum(xs*ys))-((sum(xs)*sum(ys))))/(((n*(sum(xs**2))-((sum(xs))**2))*((n*sum(ys**2)-(sum(ys)**2))))**0.5)
	ys=(ys-mean(ys))**2
	Stotal=sum(ys)
	Sres=(ys-ps)**2
	Sres=sum(Sres)
	r=1-(Sres/Stotal)

	return r


m,c=best_fit_slope_and_intercept(xs,ys)

regression_line=[(m*x)+c for x in xs]

plt.scatter(xs,ys,color='r')
plt.plot(xs, regression_line)
#plt.show()

print(coefficient_of_determination(xs,regression_line))