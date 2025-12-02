import numpy as np
from sklearn.cluster import KMeans
from matplotlib import colormaps
from diffusion_tracker.utils.io import save_frames, save_video


def kmeans_clustering(vectors, n_clusters):
    """
    Perform K-means clustering on input vectors
    
    Args:
        vectors: numpy array of shape (b, c), where b is number of vectors and c is feature dimension
        n_clusters: int, number of clusters
    
    Returns:
        labels: numpy array of shape (b,), cluster labels for each vector (0, 1, 2, ...)
        centroids: numpy array of shape (n_clusters, c), cluster centers
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    centroids = kmeans.cluster_centers_
    
    return labels, centroids

def labels_to_color(labels, cmap="nipy_spectral"):
    colormap = colormaps.get_cmap(cmap)
    norm_labels = labels.astype(float) / labels.max()  # 归一化到 [0, 1]
    rgba = colormap(norm_labels)  # (H, W, 4) in [0,1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb

def visualize_cluster_frames(labels, cmap="nipy_spectral"):
    """
    Args:
        labels: (T, H, W) int array
        cmap: matplotlib colormap, 比如 "tab20", "nipy_spectral", "hsv"
    Returns:
        frames: (T, H, W, 3) uint8 RGB array
    """
    frames = []
    T = labels.shape[0]
    for t in range(T):
        rgb = labels_to_color(labels[t], cmap=cmap)
        frames.append(rgb)
    return np.stack(frames) # (T, H, W, 3)

if __name__ == "__main__":
    import torch
    from einops import rearrange
    import torch.nn.functional as F
    features = torch.load("/data/yty/tapvid_davis/task_00/hidden_states/step_049/layer_18.pt")
    T, H, W = 25, 30, 52
    features = rearrange(features, "() head n c -> n (head c) ") # (B, C)
    labels, centroids = kmeans_clustering(features, 5)
    labels = rearrange(labels, "(t h w) -> t h w", t=T, h=H, w=W)
    frames = visualize_cluster_frames(labels, cmap="nipy_spectral")
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
    frames = F.interpolate(frames, size=(480, 832), mode="bilinear")
    save_video(frames, "tmp/clustering/hidden_states.mp4", fps=10)
