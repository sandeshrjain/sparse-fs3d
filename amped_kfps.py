import numpy as np

def build_similarity_matrix(feature_matrix):
    """
    Example function to build a d x d similarity matrix A.
    feature_matrix: (N, d), each row is a feature vector for a point.
    Returns: A (d, d)
    """
    # Simple approach: compute correlation across feature dimensions
    # shape(feature_matrix) = (N, d)
    # shape(A) = (d, d)
    # You can use other definitions: RBF kernel, etc.
    # Here we do a naive correlation-like matrix for demonstration:
    return feature_matrix.T @ feature_matrix

def gcr_krylov_subspace(A, v_init, m=5):
    """
    Build a small Krylov subspace using a simplified GCR-like approach.
    A: (d, d) similarity matrix
    v_init: (d, ) initial vector
    m: number of subspace vectors to generate
    
    Returns: List of subspace vectors [v1, v2, ..., vm], each in R^d
    """
    subspace = []
    
    # Normalize the initial vector
    v_init = v_init / (np.linalg.norm(v_init) + 1e-8)
    subspace.append(v_init)

    for i in range(1, m):
        # Multiply by A
        w = A @ subspace[i-1]
        # GCR: orthogonalize w against all previous subspace vectors
        for j in range(i):
            alpha = np.dot(w, subspace[j])
            w = w - alpha * subspace[j]
        # Normalize
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-8:
            break  # subspace saturated or degenerate
        w = w / norm_w
        subspace.append(w)
    return subspace

def amplified_distance_3D_feature(point_3d, seed_points_3d, 
                                  point_feat, seed_feats, subspace):
    """
    Compute an 'amplified distance' combining:
      (a) 3D distance to the nearest seed
      (b) subspace-based feature separation
    point_3d: (3,) current point's XYZ
    seed_points_3d: (S, 3) coordinates of chosen seeds
    point_feat: (d,) current point's feature
    seed_feats: (S, d) features of chosen seeds
    subspace: list of subspace vectors [v1, v2, ...], each in R^d
    
    Returns a single scalar total_dist.
    """
    # 1) Standard geometric distance to closest seed
    dists_3d = np.sqrt(np.sum((seed_points_3d - point_3d)**2, axis=1))
    geom_dist = np.min(dists_3d)

    # 2) Krylov subspace-based "feature distance"
    min_subspace_score = 0.0
    for v in subspace:
        proj_current = np.dot(point_feat, v)
        proj_seeds = np.dot(seed_feats, v.reshape(-1, 1)).flatten()  # shape (S,)
        # distance in this subspace dimension is difference of projections
        sub_dist = np.min(np.abs(proj_current - proj_seeds))
        min_subspace_score += sub_dist

    # Weighted sum or product, here we do a simple sum:
    total_dist = geom_dist + min_subspace_score
    return total_dist

def krylov_fps(X, F, k=10, m=5):
    """
    Krylov Subspace-Based FPS (conceptual version).
    
    X: (N, 3) array of point coordinates
    F: (N, d) array of point features
    k: number of samples to select
    m: dimension of Krylov subspace
    
    Returns: indices of chosen points (length k)
    """
    N, d = F.shape
    
    A = build_similarity_matrix(F)
    
    # 1) Randomly pick an initial seed
    first_seed = np.random.randint(0, N)
    chosen_indices = [first_seed]

    # 2) Build initial vector v1 from the feature of that seed
    v1 = F[first_seed].copy()
    
    # 3) Generate subspace vectors via simplified GCR
    subspace_vectors = gcr_krylov_subspace(A, v1, m=m)
    
    # keep track of min-dist for each point to the chosen set
    for _ in range(k-1):
        best_index = None
        best_dist = -1.0
        # Current seeds
        current_seeds_3d = X[chosen_indices]           # (S, 3)
        current_seeds_feats = F[chosen_indices]        # (S, d)
        
        for i in range(N):
            if i in chosen_indices:
                continue
            dist_i = amplified_distance_3D_feature(
                X[i], current_seeds_3d, 
                F[i], current_seeds_feats, 
                subspace_vectors
            )
            if dist_i > best_dist:
                best_dist = dist_i
                best_index = i
        chosen_indices.append(best_index)

    return np.array(chosen_indices)

if __name__ == "__main__":
    np.random.seed(42)
    # 100 points in 3D d=3 dimensional features:
    N, d = 100, 3
    X = np.random.rand(N, 3).astype(np.float32)  # point coords
    F = np.random.rand(N, d).astype(np.float32)  # features

    chosen_pts = krylov_fps(X, F, k=10, m=5)
    print("chosen indices:", chosen_pts)
    print("number of chosen points:", len(chosen_pts))
