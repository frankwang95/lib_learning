import numpy as np
import metrics_kernels as mk
import sys

sys.path.insert(0, '/Volumes/d1/learning-lib/cython')
from learningC import floyd_warshall



#################### MISC ####################
# nondestructively computes the incidence matrix of the k-nearest-neighbors graph
def compute_knn(dist, k, d):
	knn = np.empty(shape=(dist.shape[0], dist.shape[0]))
	for i in range(dist.shape[0]):
		for j in range(i, dist.shape[0]):
			knn[i,j] = d(dist[i,:], dist[j,:])
			knn[j,i] = d(dist[i,:], dist[j,:])

	for i in range(knn.shape[0]):
		sort_map = [x for (y, x) in sorted(zip(knn[i,:], range(knn.shape[0])))]
		for j in sort_map[k + 1:]: knn[i,j] = float('inf')

	return(np.minimum(knn, np.transpose(knn)))



#################### MAIN ####################
def pca(data, target_dim):
	# demean
	mean = np.mean(data, axis=0)
	data = data - mean

	# compute eigenbasis
	sample_cv_mat = np.zeros(shape=(mean.shape[0], mean.shape[0]))
	for i in range(data.shape[0]):
		sample_cv_mat += np.outer(data[i,:], data[i,:])
	sample_cv_mat = sample_cv_mat / data.shape[0]

	# find smallest eigenalue
	cv_eig = np.linalg.eig(sample_cv_mat)
	
	ord_ind = [y for (x, y) in zip(cv_eig[0], range(len(cv_eig[0])))]
	drop_list = ord_ind[:target_dim]

	eig_bas = cv_eig[1]
	for i in drop_list: np.delete(eig_bas, i, axis=1)	

	# represent data in eigenbasis
	data = np.matmul(data, eig_bas)
	
	return(data)


def isomap(data, k, target_dim, d=mk.standard_d):
	# demean
	mean = np.mean(data, axis=0)
	data = data - mean

	# compute shortest path of knn distance matrix
	knn = compute_knn(data, k, d)
	floyd_warshall(knn)
	knn = knn ** 2

	# MCS algorithim on new distance matrix
	centering_mat = np.identity(data.shape[0]) - np.ones(knn.shape) / data.shape[0]
	gram_mat = - (centering_mat.dot(knn).dot(centering_mat)) / 2

	gram_eig = np.linalg.eig(gram_mat)

	ord_ind = [x for (y, x) in sorted(zip(gram_eig[0], range(len(gram_eig[0]))))]
	evec_mat = gram_eig[1][:,ord_ind[-target_dim:]]
	eval_mat = np.diag(gram_eig[0][ord_ind[-target_dim:]]) ** 0.5
	new_data = np.matmul(evec_mat, eval_mat)

	return(new_data)
