import tensorflow as tf 
import numpy as np 
import matplotlib 

matplotlib.use('Agg')

import matplotlib.pyplot as plt

BASE_LR = 1e-2
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


prev_lines = []
def draw_lines(line):

	for x, y in prev_lines:
		plt.plot(x, y, 'r')

	x, y = [ROI_LEFT, ROI_RIGHT], [line(ROI_LEFT), line(ROI_RIGHT)]
	next_line = plt.plot(x, y, 'g')
	plt.setp(next_line, linewidth=3)

	prev_lines.append((x, y))
			 

def main():
	data = generate_data()

	k_var = tf.Variable(tf.constant(0.), name='k')
	b_var = tf.Variable(tf.constant(0.), name='b')
	x_in = tf.placeholder(tf.float32, name="input_x", shape=[None, 1])
	y_in = tf.placeholder(tf.float32, name="groundtrooth_y", shape=[None, 1])

	func = tf.multiply(k_var, x_in) + b_var
	loss = tf.reduce_mean((func - y_in) ** 2)
	optimizer = tf.train.GradientDescentOptimizer(BASE_LR, name='GradientDescent')
	train_op = optimizer.minimize(loss)
	
	def draw_first():
		draw_data(data)
		plt.savefig('fig_first.png')
		plt.clf()

	def draw_last():
		global prev_lines
		prev_lines = []
		k, b = sess.run([k_var, b_var])
		def line(x):
			return k * x + b
		draw_data(data)
		draw_lines(line)
		plt.savefig('fig_last.png')


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		draw_first()

		for it in range(50):
			k, b = sess.run([k_var, b_var])

			def line(x):
				return k * x + b

			if it < 10:
				draw_lines(line)
				draw_data(data)
				plt.savefig('fig_{}.png'.format(it))
				plt.clf()

			def feed_dict():
				return {
					x_in: np.reshape(data[:, 0], (-1, 1)), 
					y_in: np.reshape(data[:, 1], (-1, 1)),
				}

			sess.run(train_op, feed_dict=feed_dict())
		
		draw_last()



if __name__ == '__main__':
	main()
