import numpy as np


class QuadricSurface:
	def __init__(self):
		self._coeff = None
		self._precison = None

	def make_surface(self, data):
		x = data[:, 0]
		y = data[:, 1]
		A = np.vstack([x ** 2, x, np.ones(len(x))]).T
		self._coeff, self._precison, _, _ = np.linalg.lstsq(A, y)

	def get_precision(self):
		return self._precison

	def eval(self, data):
		x = data[:, 0]
		y = data[:, 1]
		return self._coeff[0] * (x ** 2) + self._coeff[1] * x + self._coeff[2] - y

	def eval_point(self, data):
		x = data[0]
		y = data[1]
		return self._coeff[0] * (x ** 2) + self._coeff[1] * x + self._coeff[2] - y
