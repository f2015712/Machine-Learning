import tensorflow as tf 
#constructing graphs
x1=tf.constant(5)
x2=tf.constant(6)
#result is a tensor
result=tf.mul(x1,x2)
#prints the tensor
print(result)
#create a new session
sess=tf.Session()
#prints 30
#print(sess.run(result))
#similar to opening and closing a file
with tf.Session() as sess:
	print(sess.run(result))
sess.close()
