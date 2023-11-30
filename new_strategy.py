import torch
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from cuml import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from fast_pytorch_kmeans import KMeans
from sklearn.cluster import Birch
from sklearn_extra.cluster import KMedoids


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class NEW_Strategy:
    def __init__(self, images, net):
        self.images = images
        self.net = net

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

        dist_matrix = euclidean_dist(centers, embeddings)
        q_idxs = index[torch.argmin(dist_matrix, dim=1)]
        if weight:
            subset_weight = np.bincount(labels.cpu().numpy())
        else:
            subset_weight = np.ones((len(labels),))
        subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
        return q_idxs, torch.from_numpy(subset_weight).float().cuda()

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
        km = KMedoids(n_clusters=n)
        km.fit(cur_features)

        # Get indices of medoids in the subset of class 'c'
        medoids_indices = km.medoid_indices_

        # Map local indices in the class subset to global indices in the original dataset
        if weight:
            subset_weight = np.bincount(km.labels_)
        else:
            subset_weight = np.ones((km.n_clusters,))
        subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
        return medoids_indices, torch.from_numpy(subset_weight).float().cuda()

    def cluster_KMeansPlusPlus(self, n, weight=False):
        embeddings = self.get_embeddings(self.images)
        index = torch.arange(len(embeddings), device='cuda')
        kmeans = KMeans(n_clusters=n, mode='euclidean', init_method='++')
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids
        dist_matrix = euclidean_dist(centers, embeddings)
        cluster_idx = index[torch.argmin(dist_matrix, dim=1)]
        if weight:
            subset_weight = np.bincount(labels.cpu().numpy())
        else:
            subset_weight = np.ones((len(labels),))
        subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
        return cluster_idx, torch.from_numpy(subset_weight).float().cuda()

    def cluster_Agglomerative(self, n, weight=False):
        cur_features = self.get_embeddings(self.images)

        cur_features = cur_features.cpu().numpy()  # (5000, 2048)
        labels = AgglomerativeClustering(n_clusters=n).fit_predict(cur_features)

        clf = NearestCentroid()
        clf.fit(cur_features, labels)
        centroids = clf.centroids_

        cluster_idx = list()
        subset_weight = np.ones(len(set(labels))) if -1 not in set(labels) else np.ones(len(set(labels)) - 1)
        for label in range(n):
            points_idx = np.where(labels == label)
            num_of_pictures_in_cluster = len(points_idx[0])
            if num_of_pictures_in_cluster <= 1:
                continue
            select_num = int(n / int(cur_features.shape[0]) * num_of_pictures_in_cluster)

            target_points = cur_features[points_idx]
            centroid = centroids[label]
            distances = np.linalg.norm(target_points.cpu() - centroid, axis=1)
            selected_array = np.argsort(distances)
            add_idx = points_idx[0][selected_array][:select_num]
            cluster_idx = cluster_idx + list(add_idx)
            if weight:
                subset_weight[label] = len(points_idx)
        subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
        return cluster_idx, torch.from_numpy(subset_weight).float().cuda()
