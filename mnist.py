from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #导入mnist数据
#该数据集中一共有55000个样本，其中50000用于训练，5000用于验证

import tensorflow as tf

x = tf.placeholder(tf.float32, [None,784]) #28*28的图像，在使用时需要拉伸成784维的向量 , None表示任意维度，一般是min-batch的 batch size
y_ = tf.placeholder(tf.float32, [None,10]) #y为X真实的类别，问题可以看成一个10分类的问题
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#使用的模型为逻辑回归
y = tf.nn.softmax(tf.matmul(x,W) + b)

#cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

#步长0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100) #读数据
        sess.run(train_step, feed_dict={x:  batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))