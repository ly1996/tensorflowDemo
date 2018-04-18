import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    foo = tf.Variable(3, name='foo')
    bar = tf.Variable(2,name='bar')
    result = foo + bar
    initialize = tf.global_variables_initializer()

print (result) #Tensor("add:0", shape=(), dtype=int32)

with tf.Session(graph=graph) as sess:
    sess.run(initialize)
    res = sess.run(result)
print(res)  # 5