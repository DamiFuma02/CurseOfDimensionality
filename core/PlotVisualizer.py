import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from core.constants import SEED


class PlotVisualizer:
    """
    Motore grafico flessibile per l'analisi delle Architetture Latenti.
    Supporta visualizzazioni 2D e 3D dinamiche.
    """

    def __init__(self, n_components=3):
        if n_components not in [2, 3]:
            raise ValueError("n_components deve essere 2 o 3.")
        self.n_components = n_components

    def plot_training_history(self, histories, title="Training vs Validation Loss"):
        """
        Esegue il plot delle curve di loss per tutti i modelli forniti.
        histories: lista di dizionari {'name': str, 'history': dict}
        """
        plt.figure(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

        for i, entry in enumerate(histories):
            name = entry['name']
            h = entry['history']
            epochs = range(1, len(h['train_loss']) + 1)

            plt.plot(epochs, h['train_loss'], label=f'{name} (Train)',
                     linestyle='--', color=colors[i], alpha=0.7)
            plt.plot(epochs, h['val_loss'], label=f'{name} (Val)',
                     linestyle='-', color=colors[i], linewidth=2)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Epoche', fontsize=12)
        plt.ylabel('Model Loss log-scale', fontsize=12)
        plt.yscale('log')  # Utile per vedere differenze tra loss molto piccole
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return plt.gcf()

    def plot_latent_space(self, models_data, sample_images, labels, semantic_names, title="Latent Space Comparison", save_path="latent_space.png"):
        """
        Plots the 2D or 3D latent space for all models.
        """
        n_models = len(models_data)
        n_samples = len(sample_images)
        # The first model's latent space helps determine sample count for stars if needed,
        # but here we assume labels represents the full dataset.

        fig = plt.figure(figsize=(22, 10), facecolor='#fdfdfd')
        cmap = plt.cm.get_cmap('turbo', 10)

        # Grid setup
        gs = gridspec.GridSpec(1, n_models, figure=fig, left=0.05, right=0.9, wspace=0.15)

        for i, m in enumerate(models_data):
            ax_kwargs = {'projection': '3d'} if self.n_components == 3 else {}
            ax = fig.add_subplot(gs[0, i], **ax_kwargs)

            z_raw = m['latent']
            dim_orig = z_raw.shape[1]
            is_linear = m.get('is_linear', False)

            # Dimensionality Reduction Logic
            if dim_orig == self.n_components:
                z = z_raw
                tech_name = "Direct"
            elif is_linear:
                z = PCA(n_components=self.n_components).fit_transform(z_raw)
                tech_name = "PCA"
            else:
                z = TSNE(n_components=self.n_components, perplexity=30, random_state=SEED,
                         init='pca', learning_rate='auto').fit_transform(z_raw)
                tech_name = "t-SNE"

            subtitle = f"{m['name']}\n"
            if dim_orig != self.n_components:
                subtitle += f"({tech_name} {dim_orig}D -> {self.n_components}D)"

            # Plotting
            coords = [z[:, j] for j in range(self.n_components)]

            # Background points
            ax.scatter(*coords, c=labels, cmap=cmap, s=35, alpha=0.25, edgecolors='none')
            # sample points with labels
            sample_coords = [z[:n_samples, j] for j in range(self.n_components)]
            ax.scatter(*sample_coords, c='black', marker='*', s=200, edgecolors='white', linewidth=1, zorder=10)
            for idx in range(n_samples):
                point = [z[idx, j] for j in range(self.n_components)]
                ax.text(*point, f" {idx + 1}", fontsize=12, fontweight='black', zorder=11)

            # Styling
            ax.set_title(subtitle, fontweight='bold', fontsize=14, pad=15)
            ax.set_xticks([])
            ax.set_yticks([])

            if self.n_components == 3:
                ax.set_zticks([])
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=20, azim=45)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            else:
                ax.set_aspect('equal', 'datalim')
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # Colorbar Legend
        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=9))
        fig.colorbar(sm, cax=cbar_ax, ticks=range(10)).ax.set_yticklabels(semantic_names, fontweight='bold')

        plt.suptitle(title, fontsize=24, fontweight='bold', y=0.98)

        # Save and return
        plt.savefig(f"{save_path}/latent_space.png", bbox_inches='tight', dpi=300)
        return fig

    def plot_sample_reconstructions(self, models_data, original_imgs, title="Generation Comparison", save_path="reconstructions.png"):
        """
        Plots a comparison between original images and reconstructions from different models.
        """
        n_models = len(models_data)
        n_samples = len(original_imgs)

        fig = plt.figure(figsize=(22, 2 + 2 * n_models), facecolor='#fdfdfd')

        # Grid: Rows = Original + Models
        gs = gridspec.GridSpec(n_models + 1, n_samples, figure=fig, hspace=0.5, wspace=0.1)

        for j in range(n_samples):
            # 1. Plot Originals (Top Row)
            ax_orig = fig.add_subplot(gs[0, j])
            ax_orig.imshow(original_imgs[j].squeeze(), cmap='bone')
            ax_orig.set_title(f"S{j + 1}", fontsize=11, fontweight='bold')
            ax_orig.axis('off')

            if j == 0:
                ax_orig.text(-15, 14, "Originale", fontsize=12, fontweight='bold',
                             ha='right', va='center', color='#34495e')

            # 2. Plot Reconstructions (Subsequent Rows)
            for i, m in enumerate(models_data):
                ax_recon = fig.add_subplot(gs[i + 1, j])
                # Assuming MNIST-like 28x28. If different, adjust .reshape() or use .squeeze()
                recon_img = m['recon'][j].reshape(28, 28)
                ax_recon.imshow(recon_img, cmap='bone')
                ax_recon.axis('off')

                if j == 0:
                    ax_recon.text(-15, 14, m['name'], fontsize=12, fontweight='bold',
                                  ha='right', va='center', color='#34495e')

        plt.suptitle(title, fontsize=24, fontweight='bold', y=0.98)

        # Save and return
        plt.savefig(f"{save_path}/sample_imgs_reconstruction.png", bbox_inches='tight', dpi=300)
        return fig