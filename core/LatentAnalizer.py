import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr, entropy
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.feature_selection import mutual_info_classif
from skimage.metrics import structural_similarity as ssim


class LatentAnalizer:
    """
    A suite of alignment and comparison techniques for latent spaces.
    """

    @staticmethod
    def compute_reconstruction_metrics(orig, recon):
        """
        Calcola MSE e SSIM. Fondamentale per tutte le slide.
        Valori: MSE (basso è meglio), SSIM (vicino a 1 è meglio).
        """
        # Conversione in numpy
        orig_np = orig.detach().cpu().numpy() if torch.is_tensor(orig) else orig
        recon_np = recon.detach().cpu().numpy() if torch.is_tensor(recon) else recon

        # MSE
        mse = np.mean((orig_np.reshape(orig_np.shape[0], -1) -
                       recon_np.reshape(recon_np.shape[0], -1)) ** 2)

        # SSIM (Perceptual)
        orig_img = orig_np.reshape(-1, 28, 28)
        recon_img = recon_np.reshape(-1, 28, 28)
        avg_ssim = np.mean([ssim(orig_img[i], recon_img[i], data_range=1.0)
                            for i in range(len(orig_img))])

        return {"mse": mse, "ssim": avg_ssim}

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

    @staticmethod
    def compute_clustering_quality(latent_codes, labels):
        """
        Misura quanto le classi di FashionMNIST sono separate nel latente.
        """
        return silhouette_score(latent_codes, labels)

    @staticmethod
    def compute_mig(latent_codes, labels):
        """
        Slide 4: Mutual Information Gap (per beta-VAE).
        Misura se una sola dimensione latente cattura l'informazione della classe.
        """
        num_latents = latent_codes.shape[1]
        mi_scores = []
        for i in range(num_latents):
            mi = mutual_info_classif(latent_codes[:, i:i + 1], labels, random_state=42)[0]
            mi_scores.append(mi)

        mi_sorted = sorted(mi_scores, reverse=True)
        _, counts = np.unique(labels, return_counts=True)
        h_y = entropy(counts)

        return (mi_sorted[0] - mi_sorted[1]) / h_y if h_y > 0 else 0

    @staticmethod
    def get_model_complexity(model)-> int:
        """
        Slide 5: Efficienza dei parametri.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_weight_rank(model) -> int:
        """
        Slide 3: Verifica il collasso lineare.
        Se il rango è uguale alla dimensione latente, non c'è collasso.
        """
        with torch.no_grad():
            combined_w = None
            # Funziona se il modello ha un attributo .enc (Sequential)
            if hasattr(model, 'enc') and isinstance(model.enc, torch.nn.Sequential):
                for layer in model.enc:
                    if isinstance(layer, torch.nn.Linear):
                        w = layer.weight.detach().cpu().numpy()
                        combined_w = w if combined_w is None else w @ combined_w
            return np.linalg.matrix_rank(combined_w) if combined_w is not None else 0