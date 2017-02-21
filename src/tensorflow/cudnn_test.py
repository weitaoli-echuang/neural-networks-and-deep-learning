###测试一
# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))
# a = tf.constant(10)
# b = tf.constant(32)
# print(sess.run(a + b))


###测试二
import tensorflow as tf

input = tf.Variable(tf.random_normal([100, 28, 28, 1]))
filter = tf.Variable(tf.random_normal([5, 5, 1, 6]))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
out = sess.run(op)



###测试三
# import tensorflow as tf

# input = tf.Variable(tf.random_normal([100, 28, 28, 1]))
# filter = tf.Variable(tf.random_normal([5, 5, 1, 6]))

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess.run(tf.initialize_all_variables())

# op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# out = sess.run(op)
