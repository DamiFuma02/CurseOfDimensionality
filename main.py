import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from core.LatentAnalizer import LatentAnalizer
from core.DataManager import DataManager
from core.models import LinearAE, DeepAE, VAE, ConvAE, TransformerAE
from core.ModelTrainer import ModelTrainer
from core.LatentVisualizer import LatentVisualizer

# ==========================================
# CONFIGURAZIONE GLOBALE E DETERMINISMO
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

dm = DataManager(batch_size=128, limit_samples=2000)
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
latent_analizer = LatentAnalizer()

BETA_VALUES = [0.01,5,20]
assert len(BETA_VALUES) == 3


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


    # Procrustes: dimostra che z_ae è solo una rotazione di z_pca
    z_pca_aligned, proc_error = latent_analizer.procrustes(z_pca,z_ae.cpu().numpy())
    # test generation using the aligned PCA latent space
    with torch.no_grad():
        rec_pca_aligned = model.dec(torch.from_numpy(z_pca_aligned).float().to(dm.device))

    pca_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(x_flat,rec_pca)
    ae_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(x_flat,rec_ae)
    pca_align_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(x_flat,rec_pca_aligned)

    latentVisualizer.plot_comparison([
        {'name': f'PCA\nMSE: {pca_reconstr_metrics["mse"]:.4f}\nSSIM: {pca_reconstr_metrics["ssim"]:.3f}', 'latent': z_pca, 'recon': rec_pca},
        {'name': f'Linear AE (RAW)\nMSE: {ae_reconstr_metrics["mse"]:.4f}\nSSIM: {ae_reconstr_metrics["ssim"]:.3f}', 'latent': z_ae.cpu().numpy(),'recon': rec_ae.cpu().numpy()},
        {'name': f'PCA (AE Aligned)\nMSE: {pca_align_reconstr_metrics["mse"]:.4f}\nSSIM: {pca_align_reconstr_metrics["ssim"]:.3f}\nProcrustes Err: {proc_error:.2e}', 'latent': z_pca_aligned,'recon': rec_pca_aligned.cpu().numpy()}
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

    rank_sl = latent_analizer.get_weight_rank(m_shallow) # rango non puo superare latent_dim,
    rank_dl = latent_analizer.get_weight_rank(m_deep_lin) # rango non puo superare latent_dim,
    rank_dnl = latent_analizer.get_weight_rank(m_deep_nonlin)
    # confermando che la profondità senza attivazioni non aggiunge capacità espressiva.
    # Procrustes tra Shallow e Deep Linear per mostrare che sono lo stesso spazio
    z_dl_aligned, proc_error = latent_analizer.procrustes(z_dl.cpu().numpy(), z_sl.cpu().numpy())

    sl_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_sl)
    dl_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_dl)
    dnl_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_dnl)

    latentVisualizer.plot_comparison([
        {'name': f'Shallow Linear AE\nMSE: {sl_reconstr_metrics["mse"]:.4f}\nSSIM: {sl_reconstr_metrics["ssim"]:.3f}\nWeights Rank: {rank_sl}', 'latent': z_sl.cpu().numpy(),
         'recon': rec_sl.cpu().numpy()},
        {'name': f'Deep Linear AE (Shallow Aligned)\nMSE: {dl_reconstr_metrics["mse"]:.4f}\nSSIM: {dl_reconstr_metrics["ssim"]:.3f}\nWeights Rank: {rank_dl}\nProc. Err vs Shallow: {proc_error:.2e}', 'latent': z_dl_aligned,
         'recon': rec_dl.cpu().numpy()},
        {'name': f'Deep Non-Linear AE\nMSE: {dnl_reconstr_metrics["mse"]:.4f}\nSSIM: {dnl_reconstr_metrics["ssim"]:.3f}\nWeights Rank: {rank_dnl}', 'latent': z_dnl.cpu().numpy(),
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

    ae_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_ae)
    sil_ae = latent_analizer.compute_clustering_quality(z_ae.cpu().numpy(), labels_remapped)
    mig_ae = latent_analizer.compute_mig(z_ae.cpu().numpy(), labels_remapped)

    models_data = [
        {'name': f'Standard AE\nSSIM: {ae_reconstr_metrics["ssim"]:.3f}\nMIG: {mig_ae:.3f}\nSilh: {sil_ae:.3f}','latent': z_ae.cpu().numpy(), 'recon': rec_ae.cpu().numpy()},
    ]
    sil_scores = []
    ssim_scores = []
    mig_scores = []
    for i, beta in enumerate(BETA_VALUES):
        mig_scores.append(latent_analizer.compute_mig(latent_list[i], labels_remapped))
        ssim_scores.append(latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_list[i])["ssim"])
        sil_scores.append(latent_analizer.compute_clustering_quality(latent_list[i], labels_remapped))

    models_data.extend([
        {'name': f'beta={beta}VAE\nSSIM={ssim_scores[i]:.3f}\nMIG: {mig_scores[i]:.3f}\nSilh: {sil_scores[i]:.3f}', 'latent': latent_list[i],'recon': rec_list[i]} for i, beta in enumerate(BETA_VALUES)
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

    ssim_vae = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_vae)["ssim"]
    params_vae = latent_analizer.get_model_complexity(m_betavae)
    silh_vae = latent_analizer.compute_clustering_quality(mu_vae.cpu().numpy(), labels_remapped)

    ssim_conv = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_conv)["ssim"]
    params_conv = latent_analizer.get_model_complexity(m_conv)
    silh_conv = latent_analizer.compute_clustering_quality(z_conv.cpu().numpy(), labels_remapped)

    latentVisualizer.plot_comparison([
        {'name': f'MLP VAE\nSSIM: {ssim_vae:.3f}\nSilh: {silh_vae:.3f}\nParams count={params_vae}', 'latent': mu_vae.cpu().numpy(),
         'recon': rec_vae.cpu().numpy()},
        {'name': f'Conv AE\nSSIM: {ssim_conv:.3f}\nSilh: {silh_conv:.3f}\nParams count={params_conv}', 'latent': z_conv.cpu().numpy(),
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

    ssim_conv = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_conv)["ssim"]
    sil_conv =latent_analizer.compute_clustering_quality(z_conv_np, labels_remapped)
    ssim_trans = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_trans)["ssim"]
    sil_trans = latent_analizer.compute_clustering_quality(z_trans_np, labels_remapped)
    cka_val = latent_analizer.cka_score(z_conv_np, z_trans_np)

    latentVisualizer.plot_comparison([
        {'name': f'CNN\nSSIM: {ssim_conv:.3f}\nSilh: {sil_conv:.3f}\n', 'latent': z_conv_np,'recon': rec_conv.cpu().numpy()},
        {'name': f'Transformer\nSSIM: {ssim_trans:.3f}\nSilh: {sil_trans:.3f}\nCKA: {cka_val:.3f}', 'latent': z_trans_np,'recon': rec_trans.cpu().numpy()}
    ], imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names, title="Slide 6: Rappresentazioni Locali vs Globali")
    plt.show()
