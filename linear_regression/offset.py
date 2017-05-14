import matplotlib.pyplot as plt 

data = [-2, -1, 1, 2], [-2, 2/3, -2/3, 2]

def y(x):
	return 3 * x / 4

MAX = 3.5
MIN = -3.5

def draw(draw_offsets, filename):
	f = plt.figure()
	x = f.gca()
	x.set_aspect("equal")
	plt.xlim((MIN, MAX))
	# plt.ylim((-3, 3))
	plt.plot([MIN, MAX], list(map(y, [MIN, MAX])), 'r')

	if draw_offsets:
		draw_offsets()

	plt.plot(*data, 'bo')
	plt.axis('off')
	plt.savefig(filename, bbox_inches='tight', dpi=200)
	plt.clf()


def main():

	def draw_vertical():
		for x, y_true in zip(data[0], data[1]):
			line = plt.plot([x, x], [y_true, y(x)], 'g')
			plt.setp(line, linewidth=3)

	def draw_perpendicular():
		for x, y_true in zip(data[0], data[1]):

			k = - 4/3
			b = y_true - k * x

			# -4/3x + b = 3/4x
			x0 = b / (3/4 + 4/3)

			def perpend(x):
				return k * x + b

			line = plt.plot([x, x0], [y_true, perpend(x0)], 'g')
			plt.setp(line, linewidth=3)


	plt.xlim((-3, 3))
	plt.ylim((-3, 3))
	draw(None, 'fig_offset_no.png')
	draw(draw_vertical, 'fig_offset_y.png')
	draw(draw_perpendicular, 'fig_offset_h.png')

if __name__ == '__main__':
	main()