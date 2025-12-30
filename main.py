import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from skimage.metrics import structural_similarity as ssim

from core.LatentAligner import LatentAligner
from core.data import LatentDataManager
from core.models import LinearAE, DeepAE, VAE, ConvAE, TransformerAE
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

dm = LatentDataManager(batch_size=128, limit_samples=2000)
trainer = ModelTrainer(dm.device)
train_loader, eval_loader = dm.get_loaders()

imgs_eval, lbls_eval = next(iter(eval_loader))
labels_remapped = dm.remap_labels(lbls_eval)
N_SAMPLES = 8

MODELS_CACHE = {}
EPOCHS = 15  # Ridotto per brevità, usa 20+ per risultati ottimali
LATENT_SPACE_DIM = 3
N_COMPONENTS = 3
assert N_COMPONENTS <= LATENT_SPACE_DIM

latentVisualizer = LatentVisualizer(n_components=N_COMPONENTS)
aligner = LatentAligner()

BETA_VALUES = [0.01,5,20]
assert len(BETA_VALUES) == 3

def compute_batch_ssim(orig, recon)-> float:
    orig = orig.detach().cpu().numpy().reshape(-1, 28, 28) if torch.is_tensor(orig) else orig.reshape(-1, 28, 28)
    recon = recon.detach().cpu().numpy().reshape(-1, 28, 28) if torch.is_tensor(recon) else recon.reshape(-1, 28, 28)
    return np.mean([ssim(orig[i], recon[i], data_range=1.0) for i in range(len(orig))])


def compute_batch_mse(orig, recon) -> float:
    orig = orig.detach().cpu().numpy() if torch.is_tensor(orig) else orig
    recon = recon.detach().cpu().numpy() if torch.is_tensor(recon) else recon
    return np.mean((orig.reshape(orig.shape[0], -1) - recon.reshape(recon.shape[0], -1)) ** 2)


def get_weight_rank(model) -> int:
    with torch.no_grad():
        combined_w = None
        # Funziona per DeepAE che ha l'attributo .enc come Sequential
        for layer in model.enc:
            if isinstance(layer, torch.nn.Linear):
                w = layer.weight.detach().cpu().numpy()
                combined_w = w if combined_w is None else w @ combined_w
        return np.linalg.matrix_rank(combined_w) if combined_w is not None else 0


def get_param_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    """Slide 2: PCA vs Linear AE (Equivalenza Geometrica)"""
    print("\n--- Slide 2: PCA vs Linear AE ---")
    x_flat = imgs_eval.view(len(imgs_eval), -1).numpy()

    # PCA
    pca = PCA(n_components=LATENT_SPACE_DIM)
    z_pca = pca.fit_transform(x_flat - np.mean(x_flat, axis=0))
    rec_pca = pca.inverse_transform(z_pca) + np.mean(x_flat, axis=0)

    # Linear AE
    model = get_trained_model("linear_ae", LinearAE,
                              lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                              latent_dim=LATENT_SPACE_DIM)
    with torch.no_grad():
        rec_ae, z_ae = model(imgs_eval.to(dm.device))

    mse_pca = compute_batch_mse(x_flat, rec_pca)
    mse_ae = compute_batch_mse(x_flat, rec_ae)
    # Procrustes: dimostra che z_ae è solo una rotazione di z_pca
    z_linearae_aligned, proc_error = aligner.procrustes( z_ae.cpu().numpy(), z_pca)

    latentVisualizer.plot_comparison([
        {'name': f'PCA\nMSE: {mse_pca:.4f}', 'latent': z_pca, 'recon': rec_pca},
        {'name': f'Linear AE (RAW)\nMSE: {mse_ae:.4f}', 'latent': z_ae.cpu().numpy(),'recon': rec_ae.cpu().numpy()},
        {'name': f'Linear AE (PCA Aligned)\nProcrustes Err: {proc_error:.2e}', 'latent': z_linearae_aligned,'recon': rec_ae.cpu().numpy()}
    ], imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names, title="Slide 2: Isomorfismo PCA e Linear AE")
    plt.show()


def run_slide_3():
    """Slide 3: Collasso della Profondità (Rango e Non-Linearità)"""
    print("\n--- Slide 3: Il Collasso Lineare ---")
    m_shallow = get_trained_model("linear_ae", LinearAE,
                              lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                              latent_dim=LATENT_SPACE_DIM)
    m_deep_lin = get_trained_model("deep_linear_ae", DeepAE,
                                   lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                                   latent_dim=LATENT_SPACE_DIM, non_linear=False)
    m_deep_nonlin = get_trained_model("deep_non_linear_ae", DeepAE,
                                      lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                                      latent_dim=LATENT_SPACE_DIM, non_linear=True)

    with torch.no_grad():
        rec_sl, z_sl = m_shallow(imgs_eval.to(dm.device))
        rec_dl, z_dl = m_deep_lin(imgs_eval.to(dm.device))
        rec_dnl, z_dnl = m_deep_nonlin(imgs_eval.to(dm.device))

    rank_dl = get_weight_rank(m_deep_lin) # rango non puo superare latent_dim,
    rank_dnl = get_weight_rank(m_deep_nonlin)
    # confermando che la profondità senza attivazioni non aggiunge capacità espressiva.
    # Procrustes tra Shallow e Deep Linear per mostrare che sono lo stesso spazio
    z_dl_aligned, proc_error = aligner.procrustes(z_dl.cpu().numpy(), z_sl.cpu().numpy())
    mse_sl = compute_batch_mse(imgs_eval, rec_sl)
    mse_dl = compute_batch_mse(imgs_eval, rec_dl)
    mse_dnl = compute_batch_mse(imgs_eval, rec_dnl)

    latentVisualizer.plot_comparison([
        {'name': f'Shallow Linear AE\nMSE: {mse_sl:.4f}', 'latent': z_sl.cpu().numpy(),
         'recon': rec_sl.cpu().numpy()},
        {'name': f'Deep Linear AE (Shallow Aligned)\nMSE: {mse_dl:.4f}\nWeights Rank: {rank_dl}\nProc. Err vs Shallow: {proc_error:.2e}', 'latent': z_dl_aligned,
         'recon': rec_dl.cpu().numpy()},
        {'name': f'Deep Non-Linear AE\nMSE: {mse_dnl:.4f}\nWeights Rank: {rank_dnl}', 'latent': z_dnl.cpu().numpy(),
         'recon': rec_dnl.cpu().numpy()}
    ], imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names, title="Slide 3: Profondità vs Non-Linearità")
    plt.show()


def run_slide_4():
    """Slide 4: AE vs VAE (Continuità Topologica)"""
    print("\n--- Slide 4: AE vs VAE ---")
    m_ae = get_trained_model("deep_non_linear_ae", DeepAE, None)
    beta_vae_models = []
    for beta in BETA_VALUES:
        beta_vae_model = get_trained_model(f"betavae_{beta}", VAE,
                              lambda m, dl: trainer.train_vae(m, dl, epochs=EPOCHS, beta=beta),
                              latent_dim=LATENT_SPACE_DIM)
        beta_vae_models.append(beta_vae_model)

    rec_list = []
    latent_list = []
    with torch.no_grad():
        rec_ae, z_ae = m_ae(imgs_eval.to(dm.device))
        for i,beta in enumerate(BETA_VALUES):
            rec_vae, mu_vae, _ = beta_vae_models[i](imgs_eval.to(dm.device))
            rec_list.append(rec_vae.cpu().numpy())
            latent_list.append(mu_vae.cpu().numpy())

    sil_ae = silhouette_score(z_ae.cpu().numpy(), labels_remapped)
    sil_scores = []
    topo_scores = []
    for i, beta in enumerate(BETA_VALUES):
        sil_scores.append(silhouette_score(latent_list[i], labels_remapped))
        topo_scores.append(aligner.neighbor_correlation(z_ae.cpu().numpy(), latent_list[i]))
    models_data = [
        {'name': f'Standard AE\nSilh: {sil_ae:.3f}', 'latent': z_ae.cpu().numpy(), 'recon': rec_ae.cpu().numpy()},
    ]
    models_data.extend([
        {'name': f'beta={beta}VAE\nSilh: {sil_scores[i]:.3f}\nTopo Corr: {topo_scores[i]:.3f}', 'latent': latent_list[i],'recon': rec_list[i]} for i, beta in enumerate(BETA_VALUES)
    ])
    latentVisualizer.plot_comparison(models_data, imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names, title="Slide 4: Regolarizzazione dello Spazio Latente")
    plt.show()


def run_slide_5():
    """Slide 5: ConvAE vs MLP VAE (Bias Induttivo e CCA)"""
    print("\n--- Slide 5: ConvAE vs MLP VAE ---")
    # choosing the middle beta value
    m_betavae = get_trained_model(f"betavae_{BETA_VALUES[1]}", VAE,
                              lambda m, dl: trainer.train_vae(m, dl, epochs=EPOCHS, beta=BETA_VALUES[1]),
                              latent_dim=LATENT_SPACE_DIM)
    m_conv = get_trained_model("conv_ae", ConvAE,
                               lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                               latent_dim=LATENT_SPACE_DIM)

    with torch.no_grad():
        rec_vae, mu_vae, _ = m_betavae(imgs_eval.to(dm.device))
        rec_conv, z_conv = m_conv(imgs_eval.to(dm.device))

    ssim_conv = compute_batch_ssim(imgs_eval, rec_conv)
    params_vae = get_param_count(m_betavae)
    params_conv = get_param_count(m_conv)
    # CCA Score: quanto sono correlate le rappresentazioni Dense vs Convoluzionali?
    cca_val = aligner.cca_score(mu_vae.cpu().numpy(), z_conv.cpu().numpy())

    latentVisualizer.plot_comparison([
        {'name': f'MLP VAE\nParams: {params_vae}', 'latent': mu_vae.cpu().numpy(),
         'recon': rec_vae.cpu().numpy()},
        {'name': f'Conv AE\nParams:{params_conv}\nSSIM: {ssim_conv:.3f}\nCCA vs MLP: {cca_val:.3f}', 'latent': z_conv.cpu().numpy(),
         'recon': rec_conv.cpu().numpy()}
    ], imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names, title="Slide 5: Efficienza Spaziale delle Convoluzioni")
    plt.show()


def run_slide_6():
    """Slide 6: Transformer vs CNN (CKA e Attenzione Globale)"""
    print("\n--- Slide 6: Transformer vs CNN ---")
    m_conv = get_trained_model("conv_ae", ConvAE,
                               lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                               latent_dim=LATENT_SPACE_DIM)
    m_trans = get_trained_model("transformer_ae", TransformerAE,
                                lambda m, dl: trainer.train_standard(m, dl, epochs=EPOCHS),
                                latent_dim=LATENT_SPACE_DIM)

    with torch.no_grad():
        rec_conv, z_conv = m_conv(imgs_eval.to(dm.device))
        rec_trans, z_trans = m_trans(imgs_eval.to(dm.device))

    z_conv_np, z_trans_np = z_conv.cpu().numpy(), z_trans.cpu().numpy()

    # METRICHE CHIAVE
    # CKA: La metrica d'oro per confrontare architetture diverse (CNN vs ViT)
    cka_val = aligner.cka_score(z_conv_np, z_trans_np)
    sil_conv =silhouette_score(z_conv_np, labels_remapped)
    sil_trans = silhouette_score(z_trans_np, labels_remapped)
    ssim_trans = compute_batch_ssim(imgs_eval, rec_trans)

    latentVisualizer.plot_comparison([
        {'name': f'CNN\nSilh: {sil_conv:.3f}', 'latent': z_conv_np,
         'recon': rec_conv.cpu().numpy()},
        {'name': f'Transformer\nSilh:{sil_trans}\nCKA vs CNN: {cka_val:.3f}\nSSIM: {ssim_trans:.3f}', 'latent': z_trans_np,
         'recon': rec_trans.cpu().numpy()}
    ], imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names, title="Slide 6: Rappresentazioni Locali vs Globali")
    plt.show()
