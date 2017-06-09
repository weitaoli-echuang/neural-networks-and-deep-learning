# coding utf-8
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches


class Node:
	def __init__(self, data=None):
		self._data = data
		self._children = None
		self._center = None
		self._dim = None
		self._depth = 0

	def set_coordinates(self, center, dim):
		self._center = center
		self._dim = dim

	def get_coordinates(self):
		return self._center, self._dim

	def set_data(self, data):
		self._data = data

	def get_data(self):
		return self._data

	def get_children(self):
		return self._children

	def add_child(self, node):
		if self._children is None:
			self._children = []
		self._children.append(node)

	def data_index(self, data):
		if self._children is None:
			return None
		for child in self._children:
			if child.get_data() == data:
				return child
		return None

	def set_depth(self, depth):
		self._depth = depth

	def get_depth(self):
		return self._depth


class Tree:
	def __init__(self, data=None, split_criteria_func=None, split_func=None):
		self._root = Node(data)
		self._root.set_coordinates(np.array([0.5, 0.5]), np.array([1, 1]))
		self._split_criteria = split_criteria_func
		self._split = split_func

	def eval(self, coordinates):
		pass

	def get_root(self):
		return self._root

	def split_node(self, node, error=None):
		if self._split_criteria is None or self._split is None:
			return None
		if node.get_data() is None:
			return None
		if self._split_criteria(node, error):
			self._split(node)
			for child in node.get_children():
				self.split_node(child, error)

	def generate_tree(self, error=None):
		self.split_node(self._root, error)

	def draw_node(self, node, axis):
		center, dim = node.get_coordinates()
		left_bottom = center - dim / 2
		axis.add_patch(
			patches.Rectangle(
				(left_bottom[0], left_bottom[1]),  # (x,y)
				dim[0],  # width
				dim[1],  # height
				fill=False
			)
		)

		data = node.get_data()
		if node.get_children() is None and data is not None:
			axis.scatter(data[:, 0], data[:, 1])
			rad = np.linalg.norm(dim) / 2 * 1.30
			axis.add_patch(
				patches.Circle(xy=(center[0], center[1]), radius=rad,
				               fill=False)
			)

		if node.get_children() is not None:
			for child in node.get_children():
				self.draw_node(child, axis)

	def draw(self, axis):
		self.draw_node(self._root, axis)


def criteria(node, error=None):
	if node is None:
		return False
	if node.get_data().shape[0] > 7:
		return True
	return False


def split(node):
	if node is None:
		return None

	center, dim = node.get_coordinates()
	depth = node.get_depth()
	data = node.get_data()
	left_bottom = center - dim / 2
	child_centers = left_bottom + dim * np.array([[1 / 4, 1 / 4], [1 / 4, 3 / 4], [3 / 4, 1 / 4], [3 / 4, 3 / 4]])
	child_dim = dim / 2
	for ch_center in child_centers[:]:
		n = Node()
		n.set_coordinates(ch_center, child_dim)
		n.set_depth(depth + 1)
		ch_left_bottom = ch_center - child_dim / 2
		ch_right_top = ch_center + child_dim / 2
		temp = data[(data[:, 0] > ch_left_bottom[0]) & (data[:, 0] < ch_right_top[0])]
		ch_data = temp[(temp[:, 1] > ch_left_bottom[1]) & (temp[:, 1] < ch_right_top[1])]
		if ch_data.shape[0] == 0:
			ch_data = None
		n.set_data(ch_data)
		node.add_child(n)


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


def blend_verify():
	n1 = Node()
	n1.set_coordinates(np.array([0.25, 0.25]), np.array([0.5, 0.5]))
	n2 = Node()
	n2.set_coordinates(np.array([0.75, 0.25]), np.array([0.5, 0.5]))
	tree = Tree()
	tree.get_root().add_child(n1)
	tree.get_root().add_child(n2)

	fig = plt.figure(figsize=(20, 10))
	axis = fig.add_subplot(111, aspect='equal')
	tree.draw(axis)
	plt.show()


if __name__ == '__main__':
	data = np.array(list(zip(np.random.rand(100), np.random.rand(200))))
	x = np.linspace(0, 1, 200)
	y = x ** 2
	data = np.array([x, y]).T
	tree = Tree(data, criteria, split)
	tree.generate_tree()

	fig = plt.figure(figsize=(20, 10))
	axis = fig.add_subplot(121, aspect='equal')
	tree.draw(axis)
	fig.add_subplot(122, aspect='equal')
	plt.scatter(data[:, 0], data[:, 1])
	plt.show()

# blend_verify()
