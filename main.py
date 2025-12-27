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

# ==========================================
# CONFIGURAZIONE GLOBALE E DETERMINISMO
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Inizializzazione data manager (fisso per tutti i modelli)
dm = LatentDataManager(batch_size=128, limit_samples=2000)
trainer = ModelTrainer(dm.device)
train_loader, eval_loader = dm.get_loaders()

# Estrazione campioni fissi per la valutazione (oggettività della visualizzazione)
imgs_eval, lbls_eval = next(iter(eval_loader))
labels_remaped = dm.remap_labels(lbls_eval)

# Cache dei modelli per evitare training ridondanti e garantire confronti equi
# (Esempio: il DeepAE usato nella slide 3 deve essere lo stesso della slide 4)
MODELS_CACHE = {}

EPOCHS = 20


def get_trained_model(model_key, model_class, train_fn, **kwargs):
    """Utility per addestrare un modello una sola volta e riutilizzarlo."""
    if model_key not in MODELS_CACHE:
        print(f"\n[Training] Addestramento nuovo modello per: {model_key}")
        model = model_class(**kwargs)
        MODELS_CACHE[model_key] = train_fn(model, train_loader)
    return MODELS_CACHE[model_key]


# ==========================================
# ESECUZIONE SLIDE
# ==========================================

def run_slide_2():
    """Slide 2: PCA vs Linear AE"""
    print("\n--- Esecuzione Slide 2: Equivalenza PCA e Linear AE ---")

    # 1. PCA (Analitica)
    x_flat = imgs_eval.view(len(imgs_eval), -1).numpy()
    x_mean = np.mean(x_flat, axis=0)
    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(x_flat - x_mean)
    rec_pca = pca.inverse_transform(z_pca) + x_mean

    # 2. Linear AE (Stesso dataset)
    model = get_trained_model("linear_3d", LinearAE,
                              lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                              latent_dim=3)

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
        {'name': '2. Linear AE (Aligned)', 'latent': z_ae_aligned, 'recon': rec_ae},
        {'name': '3. Linear AE (Raw)', 'latent': z_ae, 'recon': rec_ae}
    ], imgs_eval[:8], labels_remaped, dm.semantic_names,
        title="Slide 2: L'Equivalenza Matematica tra PCA e Autoencoder Lineare")
    plt.show()


def run_slide_3():
    """Slide 3: Il Collasso Lineare"""
    print("\n--- Esecuzione Slide 3: Il Collasso Lineare ---")

    # Confrontiamo profondità senza attivazione vs con attivazione
    m_shallow = MODELS_CACHE.get("linear_3d")
    m_lin = get_trained_model("deep_linear_collapse", DeepAE,
                              lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                              latent_dim=3, non_linear=False)

    m_nonlin = get_trained_model("deep_relu", DeepAE,
                                 lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                                 latent_dim=3, non_linear=True)

    with torch.no_grad():
        rec_shallow, z_shallow = m_shallow(imgs_eval.to(dm.device))
        rec_l, z_l = m_lin(imgs_eval.to(dm.device))
        rec_n, z_n = m_nonlin(imgs_eval.to(dm.device))

    z_shallow = z_shallow.cpu().numpy()
    z_l = z_l.cpu().numpy()
    R, _ = orthogonal_procrustes(z_l, z_shallow)
    z_l_aligned = z_l @ R
    LatentVisualizer.plot_comparison([
        {'name': 'Shallow Linear', 'latent': z_shallow, 'recon': rec_shallow.cpu().numpy()},
        {'name': 'Deep Linear (Collapse)', 'latent': z_l_aligned, 'recon': rec_l.cpu().numpy()},
        {'name': 'Deep Non-Linear (MLP)', 'latent': z_n.cpu().numpy(), 'recon': rec_n.cpu().numpy()}
    ], imgs_eval[:8], labels_remaped, dm.semantic_names,
        title="Slide 3: Perché serve la Non-Linearità?\nProfondità vs Collasso.")
    plt.show()


def run_slide_4():
    """Slide 4: Regolarizzazione (VAE)"""
    print("\n--- Esecuzione Slide 4: Variational Autoencoder ---")


    beta_values = [0.1,1.2,6]
    m_vae_low = get_trained_model("vae_3d_low", VAE,
                              lambda m, dl: trainer.train_vae(m, dl, epochs=EPOCHS, beta=beta_values[0]),
                              latent_dim=3)
    m_vae_medium = get_trained_model("vae_3d_medium", VAE,
                                  lambda m, dl: trainer.train_vae(m, dl, epochs=EPOCHS, beta=beta_values[1]),
                                  latent_dim=3)
    m_vae_high = get_trained_model("vae_3d_high", VAE,
                                  lambda m, dl: trainer.train_vae(m, dl, epochs=EPOCHS, beta=beta_values[2]),
                                  latent_dim=3)

    with torch.no_grad():
        rec_low, z_low = m_vae_low(imgs_eval.to(dm.device))
        rec_vae_medium, mu_vae_medium, _ = m_vae_medium(imgs_eval.to(dm.device))
        rec_vae_high, mu_vae_high, _ = m_vae_high(imgs_eval.to(dm.device))

    LatentVisualizer.plot_comparison([
        {'name': f'beta {beta_values[0]} Standard AE (Sparse)', 'latent': z_low.cpu().numpy(), 'recon': rec_low.cpu().numpy()},
        {'name': f'beta {beta_values[1]} VAE (Gaussian)', 'latent': mu_vae_medium.cpu().numpy(), 'recon': rec_vae_medium.cpu().numpy()},
        {'name': f'beta {beta_values[2]} VAE (Gaussian)', 'latent': mu_vae_medium.cpu().numpy(), 'recon': rec_vae_medium.cpu().numpy()}
    ], imgs_eval[:8], labels_remaped, dm.semantic_names,
        title="Slide 4: Regolarizzazione dello Spazio Latente")
    plt.show()


def run_slide_5():
    """Slide 5: Inductive Bias (ConvAE)"""
    print("\n--- Esecuzione Slide 5: Convolutional AE ---")

    m_mlp = MODELS_CACHE.get("deep_relu")
    m_conv = get_trained_model("conv_ae", ConvAE,
                               lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                               latent_dim=3)

    with torch.no_grad():
        rec_mlp, z_mlp = m_mlp(imgs_eval.to(dm.device))
        rec_conv, z_conv = m_conv(imgs_eval.to(dm.device))

    LatentVisualizer.plot_comparison([
        {'name': 'MLP AE (Dense)', 'latent': z_mlp.cpu().numpy(), 'recon': rec_mlp.cpu().numpy()},
        {'name': 'Conv AE (Spatial)', 'latent': z_conv.cpu().numpy(), 'recon': rec_conv.cpu().numpy()}
    ], imgs_eval[:8], labels_remaped, dm.semantic_names,
        title="Slide 5: Inductive Bias Spaziale")
    plt.show()


def run_slide_6():
    """Slide 6: Generazione dal VAE"""
    print("\n--- Esecuzione Slide 6: Generazione di nuovi dati ---")

    m_vae = MODELS_CACHE.get("vae_3d")
    m_vae.eval()

    n = 10
    z_random = torch.randn(n * n, 3).to(dm.device)
    with torch.no_grad():
        samples = m_vae.decoder(z_random).cpu().numpy().reshape(-1, 28, 28)

    fig, axes = plt.subplots(n, n, figsize=(10, 10), facecolor='#fdfdfd')
    for i in range(n * n):
        axes[i // n, i % n].imshow(samples[i], cmap='bone')
        axes[i // n, i % n].axis('off')

    plt.suptitle("Slide 6: Campionamento dallo Spazio Latente", fontsize=16, fontweight='bold')
    plt.show()