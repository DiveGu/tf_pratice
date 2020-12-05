import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf #使用tf的v1版本
tf.disable_v2_behavior() #使用tf的v1版本

#Xu_train=np.reshape([1,2,3,4,5],[5,1])
#print(Xu_train)
A=[[1,2],[3,4],[5,6]]
a=tf.Variable(A)
bid=np.reshape([1,2,1,1],(4,1))
b=tf.nn.embedding_lookup(a,bid) 

sess=tf.Session()
sess.run(tf.global_variables_initializer())

a,b=sess.run([a,b])
print(b)
print(b.shape)
