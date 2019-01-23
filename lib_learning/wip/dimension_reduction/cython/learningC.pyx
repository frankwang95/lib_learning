import numpy as np
cimport numpy as np



# destructively computes shortest distance marix of a graph from incidence matrix
cdef floyd_warshall_c(np.ndarray[np.float64_t, ndim=2] dist):
	cdef Py_ssize_t i, j, k
	for i in range(dist.shape[0]):
		for j in range(dist.shape[0]):
			for k in range(dist.shape[0]):
				if dist[j,k] > dist[j,i] + dist[i,k]:
					dist[j,k] = dist[j,i] + dist[i,k]
	return(dist)

def floyd_warshall(graph): return floyd_warshall_c(graph)