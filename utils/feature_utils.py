import torch
from errno import EEXIST
from os import makedirs, path
import numpy as np
from typing import List, Literal

HAS_CUVS = False
HAS_SCIPY = False
try:
    import cupy as cp
    from cuvs.neighbors import ivf_flat, brute_force
    HAS_CUVS = True
except ImportError:
    pass

if not HAS_CUVS:
    try:
        from scipy.spatial import cKDTree
        HAS_SCIPY = True
    except ImportError:
        raise ImportError("Either cuvs (with cupy) or scipy is required for KNN search. Please install one of them.")

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def knn_cuvs_ivf_flat(xyz, k=16,
                      n_lists=1024,
                      metric='sqeuclidean',
                      metric_arg=2.0,
                      kmeans_n_iters=20,
                      kmeans_trainset_fraction=0.5,
                      adaptive_centers=False,
                      add_data_on_build=True,
                      conservative_memory_allocation=False,
                      n_probes=20):
    if isinstance(xyz, torch.Tensor):
        xyz_np = xyz.detach().cpu().numpy()
    elif isinstance(xyz, np.ndarray):
        xyz_np = xyz
    else:
        raise TypeError("xyz must be np.ndarray or torch.Tensor")

    xyz_gpu = cp.asarray(xyz_np, dtype=cp.float32)

    k_search = k + 1

    build_params = ivf_flat.IndexParams(
        n_lists=n_lists,
        metric=metric,
        metric_arg=metric_arg,
        kmeans_n_iters=kmeans_n_iters,
        kmeans_trainset_fraction=kmeans_trainset_fraction,
        adaptive_centers=adaptive_centers,
        add_data_on_build=add_data_on_build,
        conservative_memory_allocation=conservative_memory_allocation
    )

    index = ivf_flat.build(build_params, xyz_gpu)

    search_params = ivf_flat.SearchParams(n_probes=n_probes)

    distances, neighbors = ivf_flat.search(search_params, index, xyz_gpu, k_search)

    distances = cp.asnumpy(distances)[:, 1:]
    neighbors = cp.asnumpy(neighbors)[:, 1:]
    return neighbors, distances

def knn_cuvs_brute_force(xyz, k=16, metric="sqeuclidean"):
    if isinstance(xyz, torch.Tensor):
        xyz_np = xyz.detach().cpu().numpy()
    elif isinstance(xyz, np.ndarray):
        xyz_np = xyz
    else:
        raise TypeError("xyz must be np.ndarray or torch.Tensor")

    xyz_gpu = cp.asarray(xyz_np, dtype=cp.float32)

    k_search = k + 1

    index = brute_force.build(xyz_gpu, metric=metric)

    distances, neighbors = brute_force.search(index, xyz_gpu, k_search)

    distances = cp.asnumpy(distances)[:, 1:]
    neighbors = cp.asnumpy(neighbors)[:, 1:]

    return neighbors, distances

def fix_knn_dists_np(dists: np.ndarray) -> np.ndarray:
    row_max = np.where(np.isfinite(dists), dists, -np.inf).max(axis=1, keepdims=True)
    dists = np.where(np.isinf(dists), row_max, dists)
    return dists

def fix_knn_dists(dists: torch.Tensor) -> torch.Tensor:
    row_max = torch.where(
        torch.isfinite(dists),
        dists,
        torch.tensor(float('-inf'), device=dists.device, dtype=dists.dtype)
    ).max(dim=1, keepdim=True).values

    dists = torch.where(torch.isinf(dists), row_max, dists)
    return dists

def knn_ckdtree(xyz, k=16):
    if not HAS_SCIPY:
        raise ImportError("scipy is required for cKDTree KNN search. Please install scipy.")
    
    if isinstance(xyz, torch.Tensor):
        xyz_np = xyz.detach().cpu().numpy()
    elif isinstance(xyz, np.ndarray):
        xyz_np = xyz
    else:
        raise TypeError("xyz must be np.ndarray or torch.Tensor")
    
    tree = cKDTree(xyz_np)
    
    k_search = k + 1
    distances, neighbors = tree.query(xyz_np, k=k_search)
    
    if k == 1:
        distances = distances[:, 1:2]
        neighbors = neighbors[:, 1:2]
    else:
        distances = distances[:, 1:]
        neighbors = neighbors[:, 1:]
    
    return neighbors, distances

def fibonacci_sphere(samples, device='cpu'):
    i = torch.arange(0, samples, device=device) + 0.5
    phi = torch.acos(1 - 2 * i / samples)
    theta = torch.pi * (1 + 5 ** 0.5) * i
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)  # [samples, 3]


def compute_sh_anisotropy_loop(
    shs: torch.Tensor,
    num_dirs: int = 60,
    max_sh_degree: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    device = shs.device
    N = shs.shape[0]
    num_basis = (max_sh_degree + 1) ** 2

    dirs = fibonacci_sphere(num_dirs, device=device)  # [num_dirs, 3]
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    # [N, 3, num_basis]
    sh_view = shs.view(N, 3, num_basis)

    colors_accum = torch.zeros((N, num_dirs, 3), device=device)
    for i in range(num_dirs):
        dir_i = dirs[i].expand(N, 3)  # [N, 3]
        color_i = eval_sh(max_sh_degree, sh_view, dir_i) + 0.5  # [N, 3]
        color_i = torch.clamp(color_i, 0.0, 1.0)
        colors_accum[:, i, :] = color_i

    # [N, 1, 3]
    mean_color = colors_accum.mean(dim=1, keepdim=True)
    var = ((colors_accum - mean_color) ** 2).mean(dim=(1, 2))
    mean_energy = (mean_color ** 2).mean(dim=(1, 2))

    anisotropy_score = var / (mean_energy + eps)
    return anisotropy_score  # [N]

def compute_sh_anisotropy_loop_std(
    shs: torch.Tensor,
    num_dirs: int = 60,
    max_sh_degree: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    device = shs.device
    N = shs.shape[0]
    num_basis = (max_sh_degree + 1) ** 2

    dirs = fibonacci_sphere(num_dirs, device=device)  # [num_dirs, 3]
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    # [N, 3, num_basis]
    sh_view = shs.view(N, 3, num_basis)

    colors_accum = torch.zeros((N, num_dirs, 3), device=device)
    for i in range(num_dirs):
        dir_i = dirs[i].expand(N, 3)  # [N, 3]
        color_i = eval_sh(max_sh_degree, sh_view, dir_i) + 0.5  # [N, 3]
        color_i = torch.clamp(color_i, 0.0, 1.0)
        colors_accum[:, i, :] = color_i

    # [N, 1, 3]
    mean_color = colors_accum.mean(dim=1, keepdim=True)
    var = ((colors_accum - mean_color) ** 2).mean(dim=(1, 2))

    anisotropy_score = var.sqrt()
    return anisotropy_score  # [N]

def compute_knn_z_score(
    feature: torch.Tensor,
    knn_indices: torch.Tensor,
    pooling: Literal["mean", "max", "none"] = "none",
    eps: float = 1e-8
) -> torch.Tensor:
    if feature.ndim == 1:
        feature = feature[:, None]  # [N, 1]

    # Get KNN neighbor features: [N, K, C]
    neighbors_feat = feature[knn_indices]  # gather neighbors

    # Compute neighbor mean and std: [N, C]
    mean = neighbors_feat.mean(dim=1)
    std = neighbors_feat.std(dim=1, unbiased=False) + eps

    # Compute z-score for each feature: [N, C]
    z = (feature - mean) / std

    # Pooling
    if pooling == "none":
        return z  # [N, C]
    elif pooling == "mean":
        return z.mean(dim=1, keepdim=True)  # [N, 1]
    elif pooling == "max":
        return z.max(dim=1, keepdim=True).values  # [N, 1]
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

def z_score_tensor(
    tensor: torch.Tensor,
    cutoff: float = 4.0,
) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    elif tensor.ndim != 2:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    z_score = (tensor - mean) / std
    
    if cutoff > 0:
        z_score_clamped = z_score.clamp(min=-cutoff, max=cutoff)  # Clamp to [-cutoff, cutoff]
        z_score_normalized = (z_score_clamped + cutoff) / (2 * cutoff)
    else:
        z_score_normalized = z_score
    return z_score_normalized  # [N, C]


def percentile_cutoff_normalize(point_features, lower_percentile=0.01, upper_percentile=0.99, cut_off_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]):
    cut_off_featues = point_features[:, cut_off_idx]
    # Compute lower and upper bounds for each feature dimension
    lower = torch.quantile(cut_off_featues, lower_percentile, dim=0)
    upper = torch.quantile(cut_off_featues, upper_percentile, dim=0)
    
    # Clip values
    clipped = torch.clamp(cut_off_featues, min=lower, max=upper)
    
    # Normalize to [0,1]
    normalized = (clipped - lower) / (upper - lower)
    
    point_features[:, cut_off_idx] = normalized
    return point_features