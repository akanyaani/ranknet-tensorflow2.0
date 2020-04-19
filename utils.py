import tensorflow as tf
import numpy as np


def gelu(x):
	with tf.name_scope("gelu"):
		cdf = 0.5 * (1.0 + tf.tanh(
			(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
		return x * cdf
