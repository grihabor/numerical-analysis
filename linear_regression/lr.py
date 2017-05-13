import tensorflow as tf 
import numpy as np 
import matplotlib 

matplotlib.use('Agg')

import matplotlib.pyplot as plt

BASE_LR = 0.1
DATA_SIZE = 20

ROI_LEFT = -8
ROI_TOP = -6
ROI_BOTTOM = 20
ROI_RIGHT = 14


def generate_data():
	cov = [[1, 0],
		   [0, 1]]
	data = [np.random.multivariate_normal([0, 0], cov) for i in range(DATA_SIZE)] + \
	    [np.random.multivariate_normal([3, 6], cov) for i in range(DATA_SIZE)] + \
	    [np.random.multivariate_normal([6, 12], cov) for i in range(DATA_SIZE)]
	data = np.array(data)
	return data


def draw_data(data):
	plt.xlim((ROI_LEFT, ROI_RIGHT))
	plt.ylim((ROI_TOP, ROI_BOTTOM))
	plt.plot(data[:, 0], data[:, 1], 'bo')
	plt.savefig('t.png')


def main():
	data = generate_data()
	draw_data(data)

	k = tf.Variable(tf.constant(0.), name='k')
	b = tf.Variable(tf.constant(0.), name='b')
	x = tf.placeholder(tf.float32, name="input_x", shape=[None, 1])
	y = tf.placeholder(tf.float32, name="groundtrooth_y", shape=[None, 1])

	func = tf.multiply(k, x) + b
	loss = (func - y) ** 2
	optimizer = tf.train.GradientDescentOptimizer(BASE_LR, name='GradientDescent')

if __name__ == '__main__':
	main()
