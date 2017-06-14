# coding utf-8
import numpy as np
from matplotlib import pyplot as plt
from Tree import Tree


def weight(d, r):
	# if d > r:
	# 	return 0
	# t = 1.5 * d / r
	# if t < 0.5:
	# 	return -t ** 2 + 0.75
	# else:
	# 	return 0.5 * (1.5 - t) ** 2
	d[d > r] = 0
	t = 1.5 * d / r
	less_mask = (t < 0.5) & (t > 0)
	less = t[(t < 0.5) & (t > 0)]
	large_mask = t > 0.5
	large = t[t > 0.5]
	less = -less ** 2 + 0.75
	large = 0.5 * (1.5 - large) ** 2
	t[large_mask] = large
	t[less_mask] = less
	return t


if __name__ == '__main__':
	# data = np.array(list(zip(np.random.rand(100), np.random.rand(200))))
	x2 = np.linspace(0, 0.5, 60)
	y2 = x2 ** 2

	x1 = np.linspace(0.5, 1, 60)
	y1 = (0.5 - x1) ** 2

	x = np.hstack((x2, x1))
	y = np.hstack((y2, y1))

	# x = np.linspace(0, 1, 200)
	# y = x ** 2

	x = np.linspace(0, 2 * np.pi, 100)
	y = np.sin(x)
	noise = np.random.rand(y.shape[0]) * 0.4
	y = y + noise

	data = np.array([x, y]).T

	fig_data = plt.figure(figsize=(20, 10))
	plt.scatter(data[:, 0], data[:, 1])
	plt.show()

	tree = Tree(data, max_depth=7)
	tree.generate_tree(error=5.0e-2)

	x_max = x.max()
	x_min = x.min()
	y_max = y.max()
	y_min = y.min()

	x_v = np.linspace(x_min, x_max, 500)
	y_v = np.linspace(y_min, y_max, 100)
	x_im, y_im = np.meshgrid(x_v, y_v)
	z = np.zeros(x_im.shape)

	count = x_im.shape[0] * x_im.shape[1]
	pos = 0
	for i in np.arange(0, x_im.shape[0], 1):
		for j in np.arange(0, x_im.shape[1], 1):
			z[i][j] = tree.eval(np.array([x_im[i][j], y_im[i][j]]))
			pos = pos + 1
		print(pos / count)

	fig = plt.figure(figsize=(20, 10))
	axis = fig.add_subplot(221, aspect='equal')
	tree.draw(axis)
	tree.draw_eval_node(axis)

	fig.add_subplot(222, aspect='equal')
	plt.scatter(data[:, 0], data[:, 1])

	axis2 = plt.subplot(223, aspect='equal')
	CS = axis2.contour(x_v, y_v, z, 1, color='r')
	axis2.clabel(CS, fontsize=20, color='r')
	axis2.scatter(x, y, color='y')
	plt.show()
