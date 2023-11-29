import torch
from fast_pytorch_kmeans import KMeans
from cuml import DBSCAN
from sklearn.cluster import Birch
from sklearn_extra.cluster import KMedoids
import numpy as np


class NEW_Strategy:
    def __init__(self, images, net):
        self.images = images
        self.net = net

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def get_embeddings(self, images):
        embed = self.net.embed
        with torch.no_grad():
            features = embed(images).detach()
        return features

    def query(self, n, weight=False):

        embeddings = self.get_embeddings(self.images)

        index = torch.arange(len(embeddings), device='cuda')

        kmeans = KMeans(n_clusters=n, mode='euclidean')
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids

        dist_matrix = self.euclidean_dist(centers, embeddings)
        q_idxs = index[torch.argmin(dist_matrix, dim=1)]
        if weight:
            subset_weight = np.bincount(labels)
        else:
            subset_weight = np.ones((labels,))
        return q_idxs, torch.from_numpy(subset_weight).cuda()

    def cluster_DBSCAN(self, min_samples, eps, weight=False):
        cur_features = self.get_embeddings(self.images)
        cur_features = cur_features.cpu().numpy()

        dbscan = DBSCAN(min_samples=min_samples, eps=eps)
        dbscan.fit(cur_features)
        labels = dbscan.labels_
        cluster_idx = list()
        subset_weight = np.ones(len(set(labels))) if -1 not in set(labels) else np.ones(len(set(labels)) - 1)
        for label in np.unique(labels):
            if label == -1:
                continue
            points_idx = np.where(labels == label)
            target_points = cur_features[points_idx]
            centroid = target_points.mean(axis=0)
            distances = np.linalg.norm(target_points - centroid, axis=1)
            cluster_idx.append(points_idx[0][np.argmin(distances)])
            if weight:
                subset_weight[label] = len(points_idx)
        subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
        return cluster_idx, torch.from_numpy(subset_weight).float().cuda()

    def cluster_BIRCH(self, n, weight=False):
        cur_features = self.get_embeddings(self.images)
        cur_features = cur_features.cpu().numpy()
        birch = Birch(n_clusters=n)
        birch.fit(cur_features)
        labels = birch.labels_

        cluster_idx = list()
        subset_weight = np.ones(len(set(labels))) if -1 not in set(labels) else np.ones(len(set(labels)) - 1)
        for label in np.unique(labels):
            points_idx = np.where(labels == label)
            target_points = cur_features[points_idx]
            centroid = target_points.mean(axis=0)
            distances = np.linalg.norm(target_points - centroid, axis=1)
            cluster_idx.append(points_idx[0][np.argmin(distances)])
            if weight:
                subset_weight[label] = len(points_idx)
        subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
        return cluster_idx, torch.from_numpy(subset_weight).float().cuda()

    def cluster_kmedoids(self, n, weight=False):
        cur_features = self.get_embeddings(self.images)
        cur_features = cur_features.cpu().numpy()
        # Perform K-medoids clustering
        kmedoids = KMedoids(n_clusters=n, metric='euclidean', random_state=0)
        kmedoids.fit(cur_features)

        # Get indices of medoids in the subset of class 'c'
        medoids_indices = kmedoids.medoid_indices_

        # Map local indices in the class subset to global indices in the original dataset
        # global_indices = np.where(labels == c)[0][medoids_indices]
        if weight:
            subset_weight = np.bincount(kmedoids.labels_)
        else:
            subset_weight = np.ones((kmedoids.n_clusters,))

        return medoids_indices, torch.from_numpy(subset_weight).float().cuda()

    def cluster_KMeansPlusPlus(self, n, weight=False):
        embeddings = self.get_embeddings(self.images)
        index = torch.arange(len(embeddings), device='cuda')
        kmeans = KMeans(n_clusters=n, mode='euclidean', init_method='++')
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids
        dist_matrix = self.euclidean_dist(centers, embeddings)
        cluster_idx = index[torch.argmin(dist_matrix, dim=1)]
        if weight:
            subset_weight = np.bincount(labels)
        else:
            subset_weight = np.ones((labels,))
        return cluster_idx, torch.from_numpy(subset_weight).float().cuda()
