import numpy as np



#################### METRICS ####################
def standard_d(x, y): return(np.linalg.norm(x - y) ** (0.5))



#################### KERNELS ####################
def linear_k(x, y): return(np.dot(x, y))


def poly_k(degree, scalar, const):
	def Tpoly_k(x, y): return((scalar * np.dot(x, y) + const) ** degree)
	return(Tpoly_k)


def gaussian_k(var):
	def Tgaussian_k(x, y):
		p = - (standard_d(x, y) ** 2) / (2 * var)
		return(np.exp(p))
	return(Tgaussian_k)