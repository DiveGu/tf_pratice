import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf #使用tf的v1版本
#tf.disable_v2_behavior() #使用tf的v1版本
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## 常量和变量
## 常量值不可以改变，常量的重新赋值相当于创造新的内存空间
#c = tf.constant([1.0,2.0])
#print(c)
#print(id(c))
#c = c + tf.constant([1.0,1.0])
#print(c)
#print(id(c))

## 变量的值可以改变，可以通过assign, assign_add等方法给变量重新赋值
#v = tf.Variable([1.0,2.0],name = "v")
#print(v)
#print(id(v))
#v.assign_add([1.0,1.0])
#print(v)
#print(id(v))

## 计算图
## tf1.0 静态图
#定义计算图
#g = tf.Graph()
#with g.as_default():
#    #placeholder为占位符，执行会话时候指定填充对象
#    x = tf.placeholder(name='x', shape=[], dtype=tf.string)  
#    y = tf.placeholder(name='y', shape=[], dtype=tf.string)
#    z = tf.string_join([x,y],name = 'join',separator=' ')
##执行计算图
#with tf.Session(graph = g) as sess:
#    print(sess.run(fetches = z,feed_dict = {x:"hello",y:"world"}))

## tf2.0 静态图

#g = tf.compat.v1.Graph()
#with g.as_default():
#    x = tf.compat.v1.placeholder(name='x', shape=[], dtype=tf.string)
#    y = tf.compat.v1.placeholder(name='y', shape=[], dtype=tf.string)
#    z = tf.strings.join([x,y],name = "join",separator = " ")
#with tf.compat.v1.Session(graph = g) as sess:
#    # fetches的结果非常像一个函数的返回值，而feed_dict中的占位符相当于函数的参数序列。
#    result = sess.run(fetches = z,feed_dict = {x:"hello",y:"world"})
#    print(result)

## tf2.0 动态图
# 动态计算图在每个算子处都进行构建，构建后立即执行
x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x,y],separator=" ")
print(z)
tf.print(z)

#def strjoin(x,y):
#    z =  tf.strings.join([x,y],separator = " ")
#    tf.print(z)
#    return z
#result = strjoin(tf.constant("hello"),tf.constant("world"))
#print(result)

