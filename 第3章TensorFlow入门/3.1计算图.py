import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b

print(a.graph is tf.get_default_graph())

#在计算图g1中定义变量"v"，并设置初始值为0
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer)

#在计算图g2中定义变量"v"，并设置初始值为0
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable(name="v", shape=[1], initializer=tf.ones_initializer)

#在计算图g1中读取变量"v"的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

#在计算图g2中读取变量"v"的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))


