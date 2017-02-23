import numpy as np



class Perceptron:
	def __init__(self, with_bias = True):
		self.first_run = False
		self.train_set = None
		self.with_bias = with_bias

		self.w = 0
		self.b = 0

		self.errors = [0]


	def train(self, new_train_data):
		if not self.first_run:
			self.w = np.zeros(new_train_data.shape[1] - 1)
			self.train_set = new_train_data
			self.first_run = True
		else: self.train_set = np.concatenate((self.train_set, new_train_data))

		for i in range(new_train_data.shape[0]):
			l = self.errors[-1]

			pred = self.pred(new_train_data[i, :-1])
			if pred == 0 and new_train_data[i, -1] == 1:
				self.w += new_train_data[i, :-1]
				if self.with_bias: self.b += 1

				l += 1
			if pred == 1 and new_train_data[i, -1] == 0:
				self.w -= new_train_data[i, :-1]
				if self.with_bias: self.b -= 1

				l += 1

			self.errors.append(l)


	def pred(self, x):
		return(int(np.inner(self.w, x) + self.b >= 0))


	def train_err_rate(self):
		err = 0
		for i in range(self.train_set.shape[0]):
			if self.pred(self.train_set[i,:-1]) != self.train_set[i,-1]: err += 1
		return(float(err) / self.train_set.shape[0])