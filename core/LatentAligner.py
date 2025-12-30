import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.metrics import pairwise_distances


class LatentAligner:
    """
    A suite of alignment and comparison techniques for latent spaces.
    """

    @staticmethod
    def procrustes(source, target):
        """
        Slide 2 & 3: Geometric alignment (Rotation, Translation, Scaling).
        Measures how much 'bending' is needed to match two spaces.
        """
        mu_s, mu_t = source.mean(0), target.mean(0)
        z_s, z_t = source - mu_s, target - mu_t

        norm_s = np.linalg.norm(z_s)
        norm_t = np.linalg.norm(z_t)
        z_s_norm = z_s / (norm_s + 1e-9)
        z_t_norm = z_t / (norm_t + 1e-9)

        R, _ = orthogonal_procrustes(z_s_norm, z_t_norm)
        z_s_aligned = (z_s_norm @ R) * norm_t + mu_t

        error = np.mean(np.square(z_s_norm @ R - z_t_norm))
        return z_s_aligned, error

    @staticmethod
    def neighbor_correlation(source, target, k=15):
        """
        Slide 4: Measures Topological Continuity.
        High correlation means neighbors in space A are still neighbors in space B.
        """
        dist_s = pairwise_distances(source)
        dist_t = pairwise_distances(target)

        # Spearman correlation of distance matrices
        corr, _ = spearmanr(dist_s.flatten(), dist_t.flatten())
        return corr

    @staticmethod
    def cca_score(source, target):
        """
        Slide 5: Canonical Correlation Analysis.
        Measures how well dimensions of A can be linearly mapped to dimensions of B.
        Used to detect Disentanglement.
        """
        n_comp = min(source.shape[1], target.shape[1])
        cca = CCA(n_components=n_comp)
        s_c, t_c = cca.fit_transform(source, target)
        # Return average correlation across canonical components
        correlations = [np.corrcoef(s_c[:, i], t_c[:, i])[0, 1] for i in range(n_comp)]
        return np.mean(correlations)

    @staticmethod
    def cka_score(source, target):
        """
        Slide 6: Centered Kernel Alignment (Linear).
        The gold standard for comparing CNNs and Transformers.
        Measures structural similarity independent of orthogonal transformations.
        """

        def feature_kernel(X):
            return X @ X.T

        K = feature_kernel(source)
        L = feature_kernel(target)

        # Center kernels
        def center(K):
            n = K.shape[0]
            unit = np.ones([n, n]) / n
            return K - unit @ K - K @ unit + unit @ K @ unit

        K_c = center(K)
        L_c = center(L)

        # HSIC calculation
        hsic = np.sum(K_c * L_c)
        norm = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))
        return hsic / norm

    @staticmethod
    def active_units_ratio(latent_codes, threshold=0.01):
        """
        Specific for beta-VAE: Measures the ratio of 'active' dimensions.
        A dimension is active if its variance is above a threshold.
        """
        variances = np.var(latent_codes, axis=0)
        active_dims = np.sum(variances > threshold)
        return active_dims / latent_codes.shape[1], variances