import tensorflow as tf

session = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(session.run(adder_node, feed_dict={a:3, b:4.5}))
print(session.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))

add_and_triple = adder_node * 3
print(session.run(add_and_triple, feed_dict={a:4, b:2}))

