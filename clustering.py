from typing import List

import numpy as np
from sklearn.decomposition import PCA
import sklearn.preprocessing as skpreprocessing
import hdbscan

from classes import ImageFile
def pca_images(
        images: List[ImageFile], n_pca: int, normalize: bool
    ) -> np.ndarray: 
    """
    Return pca resutls. Each row of resulted array corresponding to each image 
    with the same order as the input list. 
    """
    image_array = np.array([im.contents for im in images])
    if normalize: 
        skpreprocessing.normalize(image_array, copy=False)
    pca = PCA(n_components=n_pca)
    pca_array = pca.fit_transform(image_array)
    return pca_array

def cluster_images(
        images: List[ImageFile], 
        n_pca: int, 
        normalize: bool, 
        min_cluster_size: int, 
        prob_thre: float, 
        n_jobs: int
    ) -> List[ImageFile]: 
    pca_array = pca_images(images, n_pca, normalize)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, core_dist_n_jobs=n_jobs
    )
    clustered = clusterer.fit(pca_array)

    for idx in range(len(images)):
        if clustered.probabilities_[idx] < prob_thre: 
            images[idx].label = -1
        else: 
            images[idx].label = clustered.labels_[idx]
        images[idx].probability = clustered.probabilities_[idx]
    
    return images