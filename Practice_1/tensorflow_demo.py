import tensorflow as tf

# Eager execution
a = tf.constant(10)
b = tf.constant(5)

print("Eager execution result:", a + b)

# Computational graph
@tf.function
def multiply(x, y):
    return x * y

print("Graph execution result:", multiply(a, b))




