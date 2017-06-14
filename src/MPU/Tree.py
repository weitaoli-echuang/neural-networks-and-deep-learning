from matplotlib import patches
import numpy as np
from Node import Node
from scipy.spatial import KDTree


class Tree:
	def __init__(self, data, max_depth=None, max_error=None):
		self._root = Node(data)
		self._root.set_tree(self)
		x = data[:, 0]
		y = data[:, 1]
		dim = np.array([x.max() - x.min(), y.max() - y.min()])
		corner = np.array([x.min(), y.min()])
		self._root.set_coordinates(corner + dim / 2., 1.2 * dim)
		self._max_depth = max_depth
		self._max_error = max_error
		self._kdTree = KDTree(data, 7)
		self._eval_node = []

	def eval(self, coordinates):
		center, dim = self._root.get_coordinates()
		left_bottom = center - dim / 2
		right_top = center + dim / 2
		above_left = coordinates - left_bottom
		below_top = coordinates - right_top
		# in boundary
		if len(above_left[above_left >= 0]) == 2 and len(below_top[below_top <= 0]) == 2:
			w, wf = self._root.eval(coordinates)
			return wf / w
		else:
			return -10000

	def get_root(self):
		return self._root

	def split_node(self, node):
		if self._max_depth is not None and node.get_depth() > self._max_depth:
			return
		# split the node
		if node.split_criteria(self._max_error):
			node.split(self)
			for child in node.get_children():
				self.split_node(child)

	def generate_tree(self, error=None):
		self._max_error = error * np.linalg.norm(self._root._dim)
		self.split_node(self._root)

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

		# leaf node
		data = node.get_data()
		rad = node.get_rad()
		if node.get_children() is None:
			if data is not None:
				axis.scatter(data[:, 0], data[:, 1])
				# rad = np.linalg.norm(dim) / 2 * 1.30
				axis.add_patch(
					patches.Circle(xy=(center[0], center[1]), radius=rad,
					               fill=False)
				)
			else:
				axis.add_patch(
					patches.Circle(xy=(center[0], center[1]), radius=rad,
					               fill=False, color='r')
				)

		if node.get_children() is not None:
			for child in node.get_children():
				self.draw_node(child, axis)

	def draw(self, axis):
		self.draw_node(self._root, axis)

	def draw_eval_node(self, axis):
		for node in self._eval_node:
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
			if data is not None:
				axis.scatter(data[:, 0], data[:, 1])

			rad = node.get_rad()
			axis.add_patch(
				patches.Circle(xy=(center[0], center[1]), radius=rad,
				               fill=False)
			)
