import numpy as np
from QuadricSurface import QuadricSurface


class Node:
	def __init__(self, data=None):
		self._data = data
		self._children = None
		self._center = None
		self._dim = None
		self._depth = 0
		self._surface = None
		self._tree = None
		self._rad = None

	def set_tree(self, tree):
		self._tree = tree

	def set_coordinates(self, center, dim):
		self._center = center
		self._dim = dim
		self._rad = 0.75 * self._dim.max() * np.sqrt(2)

	def get_rad(self):
		return self._rad

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

	def split_criteria(self, error):
		# if the node do not contain data, then it is not necessary to split it
		if self._data is None or self._depth > self._tree._max_depth:
			return False

		# make one surface to fit local data
		if self._surface is None:
			self._surface = QuadricSurface()
			self._surface.make_surface(self._data)

		def error_fun(fit_error):
			qs = self._surface
			precision = np.abs(qs.eval(self._data))
			# print(precision.max())
			return True if precision.max() > fit_error else False

		# the node should have enough points to make a surface
		return error_fun(error) and self._data.shape[0] > 7

	def split(self, tree):
		# since the node should be split, it does not contain a valid surface
		self._surface = None

		# split it
		left_bottom = self._center - self._dim / 2
		child_centers = left_bottom + self._dim * np.array(
			[[1 / 4, 1 / 4], [1 / 4, 3 / 4], [3 / 4, 1 / 4], [3 / 4, 3 / 4]])
		child_dim = self._dim / 2
		for ch_center in child_centers[:]:
			n = Node()
			n.set_coordinates(ch_center, child_dim)
			n.set_depth(self._depth + 1)

			# ch_left_bottom = ch_center - child_dim / 2
			# ch_right_top = ch_center + child_dim / 2
			# temp = self._data[(self._data[:, 0] > ch_left_bottom[0]) & (self._data[:, 0] < ch_right_top[0])]
			# ch_data = temp[(temp[:, 1] > ch_left_bottom[1]) & (temp[:, 1] < ch_right_top[1])]

			r = n.get_rad()
			p_index = self._tree._kdTree.query_ball_point(ch_center, r)
			while len(p_index) < 7:
				r = r + 0.1 * r
				p_index = self._tree._kdTree.query_ball_point(ch_center, r)
				# print(len(p_index))

			n._rad = r
			if len(p_index) == 0:
				ch_data = None
			else:
				ch_data = self._tree.get_root().get_data()[p_index]
				n._surface = QuadricSurface()
				n._surface.make_surface(ch_data)

			n.set_tree(tree)
			n.set_data(ch_data)
			self.add_child(n)

	def weight(self, d):
		if d > self._rad:
			return 0
		t = 1.5 * d / self._rad
		if t < 0.5:
			return -t ** 2 + 0.75
		else:
			return 0.5 * (1.5 - t) ** 2

	def eval(self, coordinates):
		w_total = 0
		wf_total = 0
		dis = np.linalg.norm(coordinates - self._center)
		if dis < self._rad:
			# leaf
			if self._surface is not None:
				return self.weight(dis), self._surface.eval_point(coordinates)
			# eval in child node
			else:
				if self._children is not None:
					for child in self._children:
						w, wf = child.eval(coordinates)
						w_total = w_total + w
						wf_total = wf_total + wf
				else:
					self._tree._eval_node.append(self)
		return w_total, wf_total
