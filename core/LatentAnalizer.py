import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr, entropy
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.feature_selection import mutual_info_classif
from skimage.metrics import structural_similarity as ssim
from fvcore.nn import FlopCountAnalysis
from core.constants import SEED
from core.models import VAE


class LatentAnalizer:
    """
    A suite of alignment and comparison techniques for latent spaces.
    """

    @staticmethod
    def compute_reconstruction_metrics(orig, recon):
        """
        Calculate MSE and SSIM. Fundamental for all slides.
        Values: MSE (lower is better), SSIM (closer to 1 is better)
        """
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
        Geometric alignment (Rotation, Translation, Scaling).
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
    def compute_clustering_quality(latent_codes, labels):
        """
        Measure how well the FashionMNIST classes are separated in the latent space.
        """
        return silhouette_score(latent_codes, labels)

    @staticmethod
    def compute_mig(latent_codes, labels):
        """
        Mutual Information Gap (per beta-VAE).
        Measures if a single latent dimension captures the class information.
        """
        num_latents = latent_codes.shape[1]
        mi_scores = []
        for i in range(num_latents):
            mi = mutual_info_classif(latent_codes[:, i:i + 1], labels, random_state=SEED)[0]
            mi_scores.append(mi)

        mi_sorted = sorted(mi_scores, reverse=True)
        _, counts = np.unique(labels, return_counts=True)
        h_y = entropy(counts)

        return (mi_sorted[0] - mi_sorted[1]) / h_y if h_y > 0 else 0

    @staticmethod
    def get_model_complexity(model)-> int:
        """
        Parameter efficiency.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_weight_rank(model) -> int:
        """
        Calculate the rank of the conv_enc's combined weight matrix.
        For VAEs, consider the path leading to mu.
        """
        with torch.no_grad():
            layers_weights = []
            if isinstance(model, VAE):
                encoder_blocks = [model.encoder, model.fc_mu]
            elif hasattr(model, 'enc'):
                encoder_blocks = [model.enc]
            else:
                return 0

            for block in encoder_blocks:
                for module in block.modules():
                    if isinstance(module, nn.Linear):
                        # Estraiamo il peso come numpy array
                        layers_weights.append(module.weight.detach().cpu().numpy())

            if not layers_weights:
                return 0
            combined_w = layers_weights[0]
            for next_w in layers_weights[1:]:
                combined_w = next_w @ combined_w

            return np.linalg.matrix_rank(combined_w)

    @staticmethod
    def compute_flops(model, input_size=(1, 1, 28, 28), verbose=False):
        """
        Computational efficiency analysis.
        Calculate the total number of Floating Point Operations (FLOPs) for a single forward pass.
        Fundamental for the comparison between CNNs and ViTs (Self-Attention vs. Convolution)
        """
        device = next(model.parameters()).device

        sample_input = torch.randn(input_size).to(device)

        model_state = model.training
        model.eval()

        fca = FlopCountAnalysis(model, sample_input)

        if not verbose:
            fca.unsupported_ops_warnings(False)  # Disattiva i warning per operazioni comuni

        total_flops = fca.total()

        model.train(model_state)

        return {
            "total_flops": total_flops,
            "mflops": total_flops / 1e6,  # MegaFLOPs
            "gflops": total_flops / 1e9  # GigaFLOPs
        }