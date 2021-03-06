"""
@Project   : tf-gpu
@Module    : get_variable_test.py
@Author    : Jeff [arfu.guo@gmail.com]
@Created   : 2018/12/31 上午12:03
@Desc      :
Difference between Variable and get_variable in TensorFlow
https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow
"""
import tensorflow as tf


def a_is_c_assert():
    with tf.variable_scope("one"):
        a = tf.get_variable("v", [1])  # a.name == "one/v:0"
    with tf.variable_scope("one", reuse=True):
        c = tf.get_variable("v", [1])  # c.name == "one/v:0"
    assert (a is c)  # Assertion is true, they refer to the same object.

    with tf.variable_scope("two"):
        d = tf.Variable(1, name="v", expected_shape=[1])
        e = tf.Variable(1, expected_shape=[1])
    try:
        assert (d is e)
    except AssertionError:
        print("d is not e")


def produce_a():
    with tf.variable_scope("three"):
        a = tf.get_variable("v", [1])
    print(a.name)


def produce_b():
    with tf.variable_scope("three", reuse=True):
        b = tf.get_variable("v", [1])
    print(b.name)


def constant_test():
    tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])

    tensor2 = tf.constant(-1.0, shape=[2, 3])

    tensor3 = tf.constant(shape=[2, 3])
    # it does not work，常量张量必须给初始值
    print("tensor3: ", tensor3)

    with tf.Session() as sess:
        print("tensor: ", sess.run(tensor))
        print("tensor2: ", sess.run(tensor2))


if __name__ == "__main__":
    a_is_c_assert()

    produce_a()
    produce_b()

    constant_test()
