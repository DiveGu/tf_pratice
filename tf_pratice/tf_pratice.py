#import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#import tensorflow.compat.v1 as tf #使用tf的v1版本
#tf.disable_v2_behavior() #使用tf的v1版本
import numpy as np


##数据
#x_data=np.random.rand(100).astype(np.float32)
#y_data=0.1*x_data+0.5

###创建结构##
#Weights=tf.Variable(tf.random.uniform([1],-1,1)) #w一维 随机【-1，1】
#biases=tf.Variable(tf.zeros([1])) #b一维 0
#y=Weights*x_data+biases
#loss=tf.reduce_mean(tf.square(y-y_data))#选择损失函数
#optimizer=tf.train.GradientDescentOptimizer(0.5)#选择优化器 0.5是学习速率
#train=optimizer.minimize(loss)#优化器是要最小化loss
#init=tf.initialize_all_variables()#初始化
###结构##

#sess=tf.Session()
#sess.run(init)

#for step in range(201):
#    sess.run(train)
#    if(step%20==0):
#        print(step,sess.run(Weights),sess.run(biases))


#输入 输入的size 输出的size 激活函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#l1=add_layer(xs,1,10,activation_function=tf.nn.relu) #隐藏层 10维
#prediction=add_layer(l1,10,1,activation_function=None) #输出层 1维
l1=tf.layers.dense(xs,10,tf.nn.relu)
prediction=tf.layers.dense(l1,1,None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#init=tf.initialize_all_variables() #已经不适用
init=tf.global_variables_initializer() 
sess=tf.Session()
sess.run(init)

x = np.linspace(-1, 1, 100)[:, np.newaxis] #将一行转为一列
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise   

for i in range(101):
    sess.run(train_step,feed_dict={xs:x,ys:y})
    if i%1==0:
        print(sess.run(loss,feed_dict={xs:x,ys:y}))
