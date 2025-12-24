import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize


class LatentVisualizer:
    """
    Motore grafico per l'analisi delle Architetture Latenti.
    Gestisce il confronto tra più modelli, visualizzando lo spazio 3D
    e le relative ricostruzioni.
    """

    @staticmethod
    def plot_comparison(models_data, original_imgs, labels, semantic_names, title=""):
        """
        Parametri:
        - models_data: Lista di dizionari {'name': str, 'latent': ndarray, 'recon': ndarray}
        - original_imgs: Array dei campioni originali (primi N da evidenziare)
        - labels: Etichette semantiche per la nuvola di punti (0-9)
        - semantic_names: Lista dei nomi delle classi in ordine semantico
        - title: Titolo principale della slide
        """
        n_models = len(models_data)
        n_samples = len(original_imgs)

        # 1. CONFIGURAZIONE FIGURA
        # Aumentiamo la dimensione verticale per dare spazio ai grafici 3D ingranditi
        fig = plt.figure(figsize=(20, 7 + 3.5 * n_models), facecolor='#fdfdfd')
        cmap = plt.cm.get_cmap('turbo', 10)

        # 2. GRIGLIA SUPERIORE: SPAZI LATENTI (3D)
        # bottom=0.45 dà più spazio verticale ai grafici 3D
        gs_top = gridspec.GridSpec(1, n_models, figure=fig, bottom=0.45, top=0.90, wspace=0.05)

        for i, m in enumerate(models_data):
            ax = fig.add_subplot(gs_top[0, i], projection='3d')
            z = m['latent']

            # A. Nuvola di punti (Background) - Alpha basso per vedere la struttura
            ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=labels, cmap=cmap, s=40, alpha=0.30, edgecolors='none')

            # B. Evidenziazione Campioni (Stelle)
            # Prendiamo i primi n_samples che corrispondono alle immagini sotto
            zs = z[:n_samples]
            ax.scatter(zs[:, 0], zs[:, 1], zs[:, 2],
                       c='black', marker='*', s=180, edgecolors='white', linewidth=0.8, zorder=10)

            # C. Etichette Numeriche (1, 2, 3...)
            for idx in range(n_samples):
                ax.text(zs[idx, 0], zs[idx, 1], zs[idx, 2], f" {idx + 1}",
                        fontsize=12, fontweight='black', zorder=11, color='black')

            # D. Ottimizzazione Visuale 3D
            ax.set_box_aspect((1, 1, 1))  # Cubo perfetto
            ax.dist = 8  # Zoom: valore più basso = più vicino (default 10)
            ax.view_init(elev=20, azim=45)
            ax.set_title(m['name'], fontweight='bold', fontsize=15, pad=10, color='#2c3e50')

            # Pulizia assi
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            # Rende i piani trasparenti per un look più moderno
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # 3. LEGENDA (COLORBAR) AD ALTA VISIBILITÀ
        # Creiamo un mappabile finto per forzare alpha=1.0 nella legenda
        cbar_ax = fig.add_axes([0.94, 0.50, 0.012, 0.35])  # Posizione a destra
        norm = Normalize(vmin=0, vmax=9)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_alpha(1.0)  # Colori pieni
        cbar.set_ticks(range(10))
        cbar.ax.set_yticklabels(semantic_names, fontsize=10, fontweight='bold', color='#2c3e50')
        cbar.outline.set_edgecolor('#2c3e50')
        cbar.set_label('Categorie Semantiche', fontsize=12, fontweight='bold', labelpad=15)

        # 4. GRIGLIA INFERIORE: RICOSTRUZIONI
        # top=0.38 assicura che non ci sia sovrapposizione con i grafici 3D sopra
        gs_bottom = gridspec.GridSpec(n_models + 1, n_samples, figure=fig, top=0.38, bottom=0.05, hspace=0.4,
                                      wspace=0.1)

        for j in range(n_samples):
            # Riga 0: Immagini Originali
            ax_orig = fig.add_subplot(gs_bottom[0, j])
            ax_orig.imshow(original_imgs[j].squeeze(), cmap='bone')
            ax_orig.set_title(f"S{j + 1}", fontsize=11, fontweight='bold')
            ax_orig.axis('off')
            if j == 0:
                ax_orig.text(-15, 14, "Originale", fontsize=12, fontweight='bold', ha='right', va='center',
                             color='#34495e')

            # Righe successive: Ricostruzioni per ogni modello
            for i, m in enumerate(models_data):
                ax_recon = fig.add_subplot(gs_bottom[i + 1, j])
                # Gestione shape (se l'output è flat 784 o 28x28)
                img_recon = m['recon'][j].reshape(28, 28)
                ax_recon.imshow(img_recon, cmap='bone')
                ax_recon.axis('off')
                if j == 0:
                    ax_recon.text(-15, 14, m['name'], fontsize=12, fontweight='bold', ha='right', va='center',
                                  color='#34495e')

        # 5. TITOLO FINALE
        plt.suptitle(title, fontsize=22, fontweight='bold', y=0.97, color='#1a1a1a')

        return fig

    @staticmethod
    def plot_generation_grid(samples, title="Generazione dallo Spazio Latente"):
        """
        Metodo specifico per la Slide 7 (Generazione).
        Visualizza una griglia di immagini generate casualmente.
        """
        n = int(np.sqrt(len(samples)))
        fig, axes = plt.subplots(n, n, figsize=(10, 10), facecolor='#fdfdfd')
        for i in range(n * n):
            ax = axes[i // n, i % n]
            ax.imshow(samples[i].reshape(28, 28), cmap='bone')
            ax.axis('off')

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig