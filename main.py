import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

# Import dai nostri moduli core
from core.data import LatentDataManager
from core.models import LinearAE, DeepAE, VAE, ConvAE
from core.trainer import ModelTrainer
from core.visualizer import LatentVisualizer

# Inizializzazione globale
dm = LatentDataManager(batch_size=128, limit_samples=2000)
trainer = ModelTrainer(dm.device)
train_loader, eval_loader = dm.get_loaders()

# Estrazione campioni fissi per la valutazione
imgs_eval, lbls_eval = next(iter(eval_loader))
labels_remaped = dm.remap_labels(lbls_eval)


def run_slide_2():
    """Slide 2: PCA vs Linear AE vs Aligned AE"""
    print("\n--- Esecuzione Slide 2: Equivalenza PCA e Linear AE ---")

    # 1. PCA
    x_flat = imgs_eval.view(len(imgs_eval), -1).numpy()
    x_mean = np.mean(x_flat, axis=0)
    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(x_flat - x_mean)
    rec_pca = pca.inverse_transform(z_pca) + x_mean

    # 2. Linear AE
    model = trainer.train_standard(LinearAE(latent_dim=3), train_loader, epochs=10)
    model.eval()
    with torch.no_grad():
        rec_ae, z_ae = model(imgs_eval.to(dm.device))
    z_ae = z_ae.cpu().numpy()
    rec_ae = rec_ae.cpu().numpy()

    # 3. Allineamento Procrustes
    R, _ = orthogonal_procrustes(z_ae, z_pca)
    z_ae_aligned = z_ae @ R

    LatentVisualizer.plot_comparison([
        {'name': '1. PCA', 'latent': z_pca, 'recon': rec_pca},
        {'name': '2. Linear AE (Raw)', 'latent': z_ae, 'recon': rec_ae},
        {'name': '3. Linear AE (Aligned)', 'latent': z_ae_aligned, 'recon': rec_ae}
    ], imgs_eval[:8], labels_remaped, dm.semantic_names,
        title="Slide 2: L'Equivalenza Matematica tra PCA e Autoencoder Lineare")
    plt.show()


def run_slide_3():
    """Slide 3: Il Collasso Lineare (Profondità senza attivazione)"""
    print("\n--- Esecuzione Slide 3: Il Collasso Lineare ---")

    # 1. Deep Linear (Senza ReLU)
    m_lin = trainer.train_standard(DeepAE(latent_dim=3, non_linear=False), train_loader, epochs=10)
    # 2. Deep Non-Linear (Con ReLU)
    m_nonlin = trainer.train_standard(DeepAE(latent_dim=3, non_linear=True), train_loader, epochs=10)

    with torch.no_grad():
        rec_l, z_l = m_lin(imgs_eval.to(dm.device))
        rec_n, z_n = m_nonlin(imgs_eval.to(dm.device))

    LatentVisualizer.plot_comparison([
        {'name': 'Deep Linear (Collapse)', 'latent': z_l.cpu().numpy(), 'recon': rec_l.cpu().numpy()},
        {'name': 'Deep Non-Linear (MLP)', 'latent': z_n.cpu().numpy(), 'recon': rec_n.cpu().numpy()}
    ], imgs_eval[:8], labels_remaped, dm.semantic_names,
        title="Slide 3: Perché serve la Non-Linearità?\nSenza attivazioni, la profondità collassa in un modello lineare.")
    plt.show()


def run_slide_4():
    """Slide 4: Linear vs Deep (Manifold Curvatura)"""
    print("\n--- Esecuzione Slide 4: Linear vs Deep AE ---")

    m_lin = trainer.train_standard(LinearAE(latent_dim=3), train_loader, epochs=10)
    m_deep = trainer.train_standard(DeepAE(latent_dim=3, non_linear=True), train_loader, epochs=15)

    with torch.no_grad():
        rec_l, z_l = m_lin(imgs_eval.to(dm.device))
        rec_d, z_d = m_deep(imgs_eval.to(dm.device))

    LatentVisualizer.plot_comparison([
        {'name': 'Linear AE (Rigido)', 'latent': z_l.cpu().numpy(), 'recon': rec_l.cpu().numpy()},
        {'name': 'Deep AE (Flessibile)', 'latent': z_d.cpu().numpy(), 'recon': rec_d.cpu().numpy()}
    ], imgs_eval[:8], labels_remaped, dm.semantic_names,
        title="Slide 4: Modellare il Manifold\nLa profondità permette di 'curvare' lo spazio per seguire i dati.")
    plt.show()


def run_slide_5():
    """Slide 5: Regolarizzazione (VAE)"""
    print("\n--- Esecuzione Slide 5: Variational Autoencoder ---")

    # Usiamo il Deep AE della slide precedente come confronto
    m_ae = trainer.train_standard(DeepAE(latent_dim=3), train_loader, epochs=10)
    m_vae = trainer.train_vae(VAE(latent_dim=3), train_loader, epochs=15, beta=1.2)

    with torch.no_grad():
        rec_ae, z_ae = m_ae(imgs_eval.to(dm.device))
        rec_vae, mu_vae, _ = m_vae(imgs_eval.to(dm.device))

    LatentVisualizer.plot_comparison([
        {'name': 'Standard AE (Sparse)', 'latent': z_ae.cpu().numpy(), 'recon': rec_ae.cpu().numpy()},
        {'name': 'VAE (Gaussian)', 'latent': mu_vae.cpu().numpy(), 'recon': rec_vae.cpu().numpy()}
    ], imgs_eval[:8], labels_remaped, dm.semantic_names,
        title="Slide 5: Regolarizzazione dello Spazio Latente\nIl VAE forza una distribuzione normale, eliminando i 'buchi' nel manifold.")
    plt.show()


def run_slide_6():
    """Slide 6: Inductive Bias (ConvAE)"""
    print("\n--- Esecuzione Slide 6: Convolutional AE ---")

    m_mlp = trainer.train_standard(DeepAE(latent_dim=3), train_loader, epochs=10)
    m_conv = trainer.train_standard(ConvAE(latent_dim=3), train_loader, epochs=15)

    with torch.no_grad():
        rec_mlp, z_mlp = m_mlp(imgs_eval.to(dm.device))
        rec_conv, z_conv = m_conv(imgs_eval.to(dm.device))

    LatentVisualizer.plot_comparison([
        {'name': 'MLP AE (Dense)', 'latent': z_mlp.cpu().numpy(), 'recon': rec_mlp.cpu().numpy()},
        {'name': 'Conv AE (Spatial)', 'latent': z_conv.cpu().numpy(), 'recon': rec_conv.cpu().numpy()}
    ], imgs_eval[:8], labels_remaped, dm.semantic_names,
        title="Slide 6: Inductive Bias Spaziale\nLe convoluzioni sfruttano la vicinanza dei pixel per una compressione più intelligente.")
    plt.show()


def run_slide_7():
    """Slide 7: Generazione dal VAE"""
    print("\n--- Esecuzione Slide 7: Generazione di nuovi dati ---")

    # Alleniamo un VAE (o usiamo quello della slide 5)
    m_vae = trainer.train_vae(VAE(latent_dim=3), train_loader, epochs=15)
    m_vae.eval()

    # Campionamento casuale dallo spazio latente N(0,1)
    n = 10
    z_random = torch.randn(n * n, 3).to(dm.device)
    with torch.no_grad():
        samples = m_vae.decoder(z_random).cpu().numpy().reshape(-1, 28, 28)

    fig, axes = plt.subplots(n, n, figsize=(10, 10), facecolor='#fdfdfd')
    for i in range(n * n):
        axes[i // n, i % n].imshow(samples[i], cmap='bone')
        axes[i // n, i % n].axis('off')

    plt.suptitle(
        "Slide 7: Navigare il Manifold\nGenerazione di nuovi capi di abbigliamento campionando lo spazio latente.",
        fontsize=16, fontweight='bold')
    plt.show()
