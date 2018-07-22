import  tensorflow as tf
g = tf.Graph()
with g.as_default():
    x = tf.constant(8,name='x_const')
    y = tf.constant(5,name='y_const')
    sum = tf.add(x,y,name='x_y_sum')
    z = tf.constant(4, name='z_const')
    sum = tf.add(sum, z, name='x_y_z_sum')
    a = tf.constant([5, 3, 2, 7, 1, 4])
    b = tf.constant([4, 6, 3])
    reshape_a = tf.reshape(a,[2,3]);
    reshape_b = tf.reshape(b, [3,1]);
    dice1 = tf.Variable(tf.random_uniform([10,1],minval=1,maxval=7,dtype=tf.int32))
    dice2 = tf.Variable(tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
    dice3 = dice1 + dice2
    result = tf.concat([dice1,dice2,dice3],1)
 #   paddings = tf.constant([[0, 0], [1, 0]])
 #   tf.pad(w, paddings, "CONSTANT")
    initialization = tf.global_variables_initializer()
    sum2 = tf.matmul(reshape_a,reshape_b)
    with tf.Session() as sess:

        sess.run(initialization)
      #  print(sum.eval())
      #  print(sum2.eval())
        print (result.eval())
        print('123')


