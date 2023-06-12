import tensorflow as tf
print(tf.__version__)

with tf.compat.v1.Session() as sess:
# with tf.Session() as sess:

  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b

  # Execute the graph and store the value that `e` represents in `result`.
  result = sess.run(c)
  print(result)