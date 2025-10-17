# deepseek_vl2/utils/cka_utils.py

'''import torch

def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute linear CKA between two activation matrices X, Y of shape [n_samples, feature_dim]
    using the feature-space (Frobenius) formula:
      CKA = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    where X_c, Y_c are mean-centered along the sample dimension.
    Returns a Python float.
    """
    # mean-center
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    # promote to double for stability
    Xc2 = Xc.double()
    Yc2 = Yc.double()

    # cross- and self-covariance
    Kxy = Xc2.T @ Yc2       # [F, F]
    Kxx = Xc2.T @ Xc2
    Kyy = Yc2.T @ Yc2

    num = torch.norm(Kxy, p='fro')**2
    den = torch.norm(Kxx, p='fro') * torch.norm(Kyy, p='fro')
    if den < eps:
        return 0.0
    return (num / den).item()

def compute_layer_cka(expert_acts: torch.Tensor) -> torch.Tensor:
    """
    expert_acts: Tensor of shape [n_tokens, n_experts, hidden_dim]
    Returns: Tensor [n_experts, n_experts] of CKA similarities.
    """
    n_tokens, n_experts, hidden_dim = expert_acts.shape
    sim = torch.zeros((n_experts, n_experts), dtype=torch.float32, device=expert_acts.device)

    # for each pair of experts
    for i in range(n_experts):
        Xi = expert_acts[:, i, :].float()   # [n_tokens, hidden_dim]
        for j in range(i, n_experts):
            Yj = expert_acts[:, j, :].float()
            c = linear_cka(Xi, Yj)
            sim[i, j] = sim[j, i] = c

    return sim
'''
'''

import torch

def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute linear CKA between two activation matrices X, Y of shape [n_samples, feature_dim]
    using the feature-space (Frobenius) formula:
      CKA = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    where X_c, Y_c are mean-centered along the sample dimension.
    Returns a Python float.
    """
    # mean-center
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    # promote to double for stability
    Xc2 = Xc.double()
    Yc2 = Yc.double()

    # cross- and self-covariance
    Kxy = Xc2.T @ Yc2       # [F, F]
    Kxx = Xc2.T @ Xc2
    Kyy = Yc2.T @ Yc2

    num = torch.norm(Kxy, p='fro')**2
    den = torch.norm(Kxx, p='fro') * torch.norm(Kyy, p='fro')
    if den < eps:
        return 0.0
    return (num / den).item()


def compute_layer_cka(expert_acts: torch.Tensor) -> torch.Tensor:
    """
    expert_acts: Tensor of shape [n_tokens, n_experts, hidden_dim]
    Returns: Tensor [n_experts, n_experts] of CKA similarities.
    """
    n_tokens, n_experts, hidden_dim = expert_acts.shape
    sim = torch.zeros((n_experts, n_experts), dtype=torch.float32, device=expert_acts.device)

    # for each pair of experts
    for i in range(n_experts):
        Xi = expert_acts[:, i, :].float()   # [n_tokens, hidden_dim]
        for j in range(i, n_experts):
            Yj = expert_acts[:, j, :].float()
            c = linear_cka(Xi, Yj)
            sim[i, j] = sim[j, i] = c

    return sim


def save_cka_heatmaps(
    cka_matrices: dict,
    output_dir: str = "cka_heatmaps",
    cmap: str = "viridis",
    figsize: tuple = (8, 6)
) -> None:
    """
    Save heatmaps of CKA similarity matrices for each MoE layer.

    Args:
        cka_matrices (dict): mapping layer index (int) to a 2D array or torch.Tensor of shape [E, E].
        output_dir (str): directory to save heatmap PNGs.
        cmap (str): matplotlib colormap name.
        figsize (tuple): figure size for each heatmap.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    for layer_idx, mat in cka_matrices.items():
        # convert torch.Tensor to numpy if needed
        try:
            if isinstance(mat, torch.Tensor):
                mat = mat.detach().cpu().numpy()
        except NameError:
            pass

        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(mat, interpolation='nearest', aspect='equal', cmap=cmap)
        ax.set_title(f"Layer {layer_idx} Expert CKA")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Expert index")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"cka_layer_{layer_idx}.png")
        fig.savefig(out_path)
        plt.close(fig)

    print(f"Saved CKA heatmaps to '{output_dir}'")'''
  
'''   
    
import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute linear CKA between two activation matrices X, Y of shape [n_samples, feature_dim]
    using the feature-space (Frobenius) formula:
      CKA = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    where X_c, Y_c are mean-centered along the sample dimension.
    Returns a Python float.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    Xc2 = Xc.double()
    Yc2 = Yc.double()
    Kxy = Xc2.T @ Yc2
    Kxx = Xc2.T @ Xc2
    Kyy = Yc2.T @ Yc2
    num = torch.norm(Kxy, p='fro')**2
    den = torch.norm(Kxx, p='fro') * torch.norm(Kyy, p='fro')
    if den < eps:
        return 0.0
    return (num / den).item()


def compute_layer_cka(expert_acts: torch.Tensor) -> torch.Tensor:
    """
    expert_acts: Tensor of shape [n_tokens, n_experts, hidden_dim]
    Returns: Tensor [n_experts, n_experts] of CKA similarities.
    """
    n_tokens, n_experts, hidden_dim = expert_acts.shape
    sim = torch.zeros((n_experts, n_experts), dtype=torch.float32, device=expert_acts.device)

    for i in range(n_experts):
        Xi = expert_acts[:, i, :].float()
        for j in range(i, n_experts):
            Yj = expert_acts[:, j, :].float()
            c = linear_cka(Xi, Yj)
            sim[i, j] = sim[j, i] = c

    return sim


def find_similar_experts(
    cka_matrices: dict,
    threshold: float = 0.7
) -> dict:
    """
    Identify expert pairs in each layer whose CKA similarity is >= threshold.

    Args:
        cka_matrices (dict): mapping layer index to a 2D torch.Tensor or numpy array of shape [E, E].
        threshold (float): similarity threshold between 0 and 1.

    Returns:
        dict: mapping layer index to a list of tuples (i, j, similarity) for i < j.
    """
    similar = {}
    for layer_idx, mat in cka_matrices.items():
        # ensure torch tensor
        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat)
        E = mat.size(0)
        pairs = []
        for i in range(E):
            for j in range(i+1, E):
                sim = mat[i, j].item()
                if sim >= threshold:
                    pairs.append((i, j, sim))
        similar[layer_idx] = pairs
    return similar


def save_cka_heatmaps(
    cka_matrices: dict,
    output_dir: str = "cka_heatmaps",
    cmap: str = "viridis",
    figsize: tuple = (8, 6)
) -> None:
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for layer_idx, mat in cka_matrices.items():
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(mat, interpolation='nearest', aspect='equal', cmap=cmap)
        ax.set_title(f"Layer {layer_idx} Expert CKA")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Expert index")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"cka_layer_{layer_idx}.png")
        fig.savefig(out_path)
        plt.close(fig)

    print(f"Saved CKA heatmaps to '{output_dir}'")

'''

'''

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def rbf_kernel(X: torch.Tensor, sigma: float = None) -> torch.Tensor:
    """
    Compute the RBF (Gaussian) kernel matrix for X.
    X: [n_samples, feature_dim]
    Returns: K [n_samples, n_samples]
    """
    # Pairwise squared Euclidean distances
    X_norm = (X**2).sum(dim=1, keepdim=True)
    dists = X_norm + X_norm.T - 2 * (X @ X.T)

    # Median heuristic for sigma
    if sigma is None:
        # Use median of distances as bandwidth
        d = dists.detach().cpu().numpy()
        sigma = np.sqrt(0.5 * np.median(d[d > 0]))
        sigma = float(sigma) if sigma > 0 else 1.0

    gamma = 1.0 / (2 * sigma**2)
    K = torch.exp(-gamma * dists)
    return K


def rbf_cka(X: torch.Tensor, Y: torch.Tensor, sigma: float = None, eps: float = 1e-8) -> float:
    """
    Compute RBF CKA between X and Y using kernel CKA formula:
      HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    where HSIC = trace(K_c L_c) and K_c, L_c are centered kernel matrices.
    Returns a Python float.
    """
    # Compute RBF kernels
    K = rbf_kernel(X, sigma)
    L = rbf_kernel(Y, sigma)

    # Centering
    n = K.size(0)
    H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
    Kc = H @ K @ H
    Lc = H @ L @ H

    # HSIC
    hsic_xy = torch.trace(Kc @ Lc)
    hsic_xx = torch.trace(Kc @ Kc)
    hsic_yy = torch.trace(Lc @ Lc)

    # Normalize
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < eps:
        return 0.0
    return float(hsic_xy / denom)


def compute_layer_cka(expert_acts: torch.Tensor, use_rbf: bool = True, sigma: float = None) -> torch.Tensor:
    """
    expert_acts: Tensor [n_tokens, n_experts, hidden_dim]
    Returns: Tensor [n_experts, n_experts] of CKA similarities.
    Set use_rbf=True to use RBF CKA.
    """
    n_tokens, n_experts, hidden_dim = expert_acts.shape
    sim = torch.zeros((n_experts, n_experts), dtype=torch.float32, device=expert_acts.device)

    # flatten token dimension for kernel
    for i in range(n_experts):
        Xi = expert_acts[:, i, :].float()
        for j in range(i, n_experts):
            Yj = expert_acts[:, j, :].float()
            if use_rbf:
                c = rbf_cka(Xi, Yj, sigma=sigma)
            else:
                # fallback to linear CKA
               # c = linear_cka(Xi, Yj)
                c = rbf_cka(Xi, Yj, sigma=sigma)
            sim[i, j] = sim[j, i] = c

    return sim


def save_cka_heatmaps(
    cka_matrices: dict,
    threshold: float = 0.7,
    output_dir: str = "cka_heatmaps",
    cmap: str = "viridis",
    figsize: tuple = (8, 6)
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for layer_idx, mat in cka_matrices.items():
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(mat, interpolation='nearest', aspect='equal', cmap=cmap)
        ax.set_title(f"Layer {layer_idx} Expert CKA")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Expert index")
        E = mat.shape[0]
        for i in range(E):
            for j in range(E):
                if i < j and mat[i, j] >= threshold:
                    ax.plot(j, i, 'x', markersize=10, markeredgewidth=2, color='red')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"cka_layer_{layer_idx}.png"))
        plt.close(fig)
    print(f"Saved CKA heatmaps to '{output_dir}', pairs = {threshold} annotated.")


def find_similar_experts(
    cka_matrices: dict,
    threshold: float = 0.7
) -> dict:
    similar = {}
    for layer_idx, mat in cka_matrices.items():
        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat)
        E = mat.size(0)
        pairs = []
        for i in range(E):
            for j in range(i+1, E):
                sim_val = mat[i, j].item()
                if sim_val >= threshold:
                    pairs.append((i, j, sim_val))
        similar[layer_idx] = pairs
    return similar

# The linear_cka function is required as fallback for compute_layer_cka

'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def rbf_kernel(X: torch.Tensor, sigma: float = None) -> torch.Tensor:
    """
    Compute the RBF (Gaussian) kernel matrix for X.
    X: [n_samples, feature_dim]
    Returns: K [n_samples, n_samples]
    """
    # Pairwise squared Euclidean distances
    X_norm = (X ** 2).sum(dim=1, keepdim=True)
    dists = X_norm + X_norm.T - 2 * (X @ X.T)

    # Median heuristic for sigma
    if sigma is None:
        d = dists.detach().cpu().numpy()
        sigma = np.sqrt(0.5 * np.median(d[d > 0]))
        sigma = float(sigma) if sigma > 0 else 1.0

    gamma = 1.0 / (2 * sigma ** 2)
    K = torch.exp(-gamma * dists)
    return K


def rbf_cka(X: torch.Tensor, Y: torch.Tensor, sigma: float = None, eps: float = 1e-8) -> float:
    """
    Compute RBF CKA similarity between tensors X and Y.
    Returns a scalar similarity score in [0, 1].
    """
    K = rbf_kernel(X, sigma)
    L = rbf_kernel(Y, sigma)

    n = K.size(0)
    H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic_xy = torch.trace(Kc @ Lc)
    hsic_xx = torch.trace(Kc @ Kc)
    hsic_yy = torch.trace(Lc @ Lc)

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < eps:
        return 0.0
    return float(hsic_xy / denom)


def compute_layer_cka(expert_acts: torch.Tensor, sigma: float = None) -> torch.Tensor:
    """
    Compute expert-to-expert RBF CKA similarity matrix for a single layer.
    expert_acts: [n_tokens, n_experts, hidden_dim]
    Returns: [n_experts, n_experts] CKA similarity matrix
    """
    n_tokens, n_experts, hidden_dim = expert_acts.shape
    sim = torch.zeros((n_experts, n_experts), dtype=torch.float32, device=expert_acts.device)

    for i in range(n_experts):
        Xi = expert_acts[:, i, :].float()
        for j in range(i, n_experts):
            Yj = expert_acts[:, j, :].float()
            cka_val = rbf_cka(Xi, Yj, sigma=sigma)
            sim[i, j] = sim[j, i] = cka_val

    return sim


def save_cka_heatmaps(
    cka_matrices: dict,
    threshold: float = 0.7,
    output_dir: str = "cka_heatmaps",
    cmap: str = "viridis",
    figsize: tuple = (8, 6)
) -> None:
    """
    Save CKA similarity heatmaps with optional threshold-based annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    for layer_idx, mat in cka_matrices.items():
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(mat, interpolation='nearest', aspect='equal', cmap=cmap)
        ax.set_title(f"Layer {layer_idx} Expert CKA")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Expert index")

        E = mat.shape[0]
        for i in range(E):
            for j in range(E):
                if i < j and mat[i, j] >= threshold:
                    ax.plot(j, i, 'x', markersize=10, markeredgewidth=2, color='red')

        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"cka_layer_{layer_idx}.png"))
        plt.close(fig)
    print(f"Saved CKA heatmaps to '{output_dir}', pairs = {threshold} annotated.")


def find_similar_experts(
    cka_matrices: dict,
    threshold: float = 0.7
) -> dict:
    """
    Return a dictionary of similar expert index pairs for each layer.
    """
    similar = {}
    for layer_idx, mat in cka_matrices.items():
        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat)

        E = mat.size(0)
        pairs = []
        for i in range(E):
            for j in range(i + 1, E):
                sim_val = mat[i, j].item()
                if sim_val >= threshold:
                    pairs.append((i, j, sim_val))

        similar[layer_idx] = pairs
    return similar
   
