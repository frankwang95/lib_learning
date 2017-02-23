import numpy as np
import matplotlib.pyplot as plt



############# HELPER FUNCTIONS #############
def d(x, y):
	return(np.linalg.norm(x - y))


# returns list with shortest distance from start to all points
def floyd_warshall(dist):
	for k in range(dist.shape[0]):
		for i in range(dist.shape[0]):
			for j in range(dist.shape[0]):
				if dist[i,j] > dist[i,k] + dist[k,j]:
					dist[i,j] = dist[i,k] + dist[k,j]
	return(dist)


def pca(data, k):
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
	drop_list = ord_ind[:k]

	eig_bas = cv_eig[1]
	for i in drop_list: np.delete(eig_bas, i, axis=1)	

	# represent data in eigenbasis
	data = np.matmul(data, eig_bas)
	
	return(data)


def isomap(data, k, p):
	# demean
	mean = np.mean(data, axis=0)
	data = data - mean

	print data

	# computing k-nn shortest path distances
	knn = np.empty(shape=(data.shape[0], data.shape[0]))
	for i in range(data.shape[0]):
		for j in range(i, data.shape[0]):
			knn[i,j] = d(data[i,:], data[j,:])
			knn[j,i] = d(data[i,:], data[j,:])

	for i in range(knn.shape[0]):
		sort_map = [x for (y, x) in sorted(zip(knn[i,:], range(knn.shape[0])))]
		for j in sort_map[k + 1:]: knn[i,j] = float('inf')

	print knn

	knn = np.minimum(knn, np.transpose(knn))
	dist = floyd_warshall(knn)

	print dist

	# MCS algorithim on new distance matrix
	centering_mat = np.identity(data.shape[0]) - np.ones(dist.shape) / data.shape[0]
	gram_mat = - np.matmul(np.matmul(centering_mat, dist), centering_mat) / 2

	gram_eig = np.linalg.eig(gram_mat)

	ord_ind = [x for (y, x) in sorted(zip(gram_eig[0], range(len(gram_eig[0]))))]
	evec_mat = gram_eig[1][:,ord_ind[-p:]]
	eval_mat = np.diag(gram_eig[0][ord_ind[-p:]]) ** 0.5
	new_data = np.matmul(evec_mat, eval_mat)

	return(new_data)

	
def data_import_3D():
	h = open('3Ddata.txt', 'r')
	data = h.readlines()
	h.close()

	data = [x.strip().split(' ') for x in data]
	data = np.array(data, dtype="float")

	return(data)


def test_isomap(data):
	r = data[:, 3]==1
	b = data[:, 3]==2
	g = data[:, 3]==3
	y = data[:, 3]==4
	
	flat = isomap(data[:,:3], 5, 2)

	plt.scatter(flat[r,0], flat[r, 1], color='red')
	plt.scatter(flat[b,0], flat[b, 1], color='blue')
	plt.scatter(flat[g,0], flat[g, 1], color='green')
	plt.scatter(flat[y,0], flat[y, 1], color='yellow')
	plt.savefig("isomap.pdf")


data = data_import_3D()[:10]
test_isomap(data)