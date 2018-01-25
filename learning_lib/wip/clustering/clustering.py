import numpy as np
import random



class KMeans:
	def __init__(self, data, n_clusters, initialization='kmeans++'):
		self.data = data
		self.n_clusters = n_clusters
		self.n_data = data.shape[0]
		self.n_dim = data.shape[1]
		self.initialization = initialization

		self.choose_initial_centroids()
		self.clusters = [[] for _ in range(self.n_clusters)]
		self.last_iteration = None

		self.n_iterations = 0

		while not self.termination_condition():
			self.iter_cluster()
			self.iter_centroid()
			self.n_iterations += 1


	##### Initialization Methods #####
	def choose_initial_centroids(self):
		if self.initialization == 'forgy': self.forgy()
		if self.initialization == 'kmeans++': self.kmpp()


	def forgy(self):
		indexes = random.sample(range(self.n_data), self.n_clusters)
		self.centroids = [self.data[i, :] for i in indexes]
		return(0)


	def kmpp(self):
		index = random.choice(range(self.n_data))
		self.centroids = [self.data[index, :]]
	
		while len(self.centroids) < self.n_clusters:
			prob = []
			for i in range(self.n_data):
				closest_centroid = (float("inf"), 0)
				for j in range(len(self.centroids)):
					d = np.linalg.norm(self.data[i, :] - self.centroids[j])**2
					if d < closest_centroid[0]: closest_centroid = (d, j)
				prob.append(closest_centroid[0])

			prob = np.array(prob) / sum(prob)
			index = np.random.choice(range(self.n_data), p=prob)
			self.centroids.append(self.data[index, :])


	##### Primary Functions ######
	def iter_cluster(self):
		self.last_iteration = self.clusters
		self.clusters = [[] for _ in range(self.n_clusters)]

		for i in range(self.n_data):
			closest_centroid = (float("inf"), 0)
			for j in range(self.n_clusters):
				d = np.linalg.norm(self.data[i, :] - self.centroids[j])**2
				if d < closest_centroid[0]: closest_centroid = (d, j)

			self.clusters[closest_centroid[1]].append(i)
		return(0)


	def iter_centroid(self):
		for i in range(self.n_clusters):
			self.centroids[i] = sum([self.data[j, :] for j in self.clusters[i]]) / len(self.clusters[i])
		return(0)


	def termination_condition(self):
		return(self.last_iteration == self.clusters)