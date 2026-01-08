import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from core.LatentAnalizer import LatentAnalizer
from core.DataManager import DataManager
from core.models import LinearAE, DeepAE, VAE, ConvAE, TransformerAE
from core.ModelTrainer import ModelTrainer
from core.PlotVisualizer import PlotVisualizer
from core.constants import SEED, N_COMPONENTS, LATENT_SPACE_DIM, EPOCHS, STATIC_ROOT

# ==========================================
# CONFIGURAZIONE GLOBALE E DETERMINISMO
# ==========================================
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


plotVisualizer = PlotVisualizer(n_components=N_COMPONENTS)
latent_analizer = LatentAnalizer()

BETA_VALUES = [0.01,5,20]
assert len(BETA_VALUES) == 3


def get_trained_model(model_key, model_class, train_fn, **kwargs):
    """Utility aggiornata per salvare modello e cronologia loss."""
    if model_key not in MODELS_CACHE:
        print(f"\n[Training] Addestramento nuovo modello per: {model_key}")
        model = model_class(**kwargs)
        model, history = train_fn(model, train_loader, eval_loader)
        MODELS_CACHE[model_key] = {'model': model, 'history': history}
    return MODELS_CACHE[model_key]['model'], MODELS_CACHE[model_key]['history']


# ==========================================
# ESECUZIONE SLIDE
# ==========================================

def run_slide_2():
    """Slide 2: Linearità vs Collasso della Profondità (Rango e Non-Linearità)"""
    x_flat = imgs_eval.view(len(imgs_eval), -1).numpy()

    pca = PCA(n_components=LATENT_SPACE_DIM)
    z_pca = pca.fit_transform(x_flat - np.mean(x_flat, axis=0))
    rec_pca = pca.inverse_transform(z_pca) + np.mean(x_flat, axis=0)

    m_shallow, sl_loss_hist = get_trained_model("linear_ae", LinearAE,
                              lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                              latent_dim=LATENT_SPACE_DIM)
    m_deep_lin, dl_loss_hist = get_trained_model("deep_linear_ae", DeepAE,
                                   lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                                   latent_dim=LATENT_SPACE_DIM, non_linear=False)
    m_deep_nonlin, dnl_loss_hist = get_trained_model("deep_non_linear_ae", DeepAE,
                                      lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
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
    z_pca_sl_aligned, pca_sl_proc_error = latent_analizer.procrustes(z_pca, z_sl.cpu().numpy())
    z_dl_sl_aligned, dl_sl_proc_error = latent_analizer.procrustes(z_dl.cpu().numpy(), z_sl.cpu().numpy())

    pca_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(imgs_eval,rec_pca)
    sl_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_sl)
    dl_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_dl)
    dnl_reconstr_metrics = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_dnl)
    models_data = [
        {   'name': 'PCA',
            'latent': z_pca_sl_aligned,
            'recon': rec_pca,
            "metrics": f'MSE: {pca_reconstr_metrics["mse"]:.4f}\nSSIM: {pca_reconstr_metrics["ssim"]:.3f}\nProc. Err vs Shallow: {pca_sl_proc_error:.2e}'},
        {   'name': 'Shallow Linear AE',
            'latent': z_sl.cpu().numpy(),
            'recon': rec_sl.cpu().numpy(),
            "metrics":f'MSE: {sl_reconstr_metrics["mse"]:.4f}\nSSIM: {sl_reconstr_metrics["ssim"]:.3f}\nWeights Rank: {rank_sl}'},
        {   'name': 'Deep Linear AE',
            'latent': z_dl_sl_aligned,
            'recon': rec_dl.cpu().numpy(),
            "metrics": f'MSE: {dl_reconstr_metrics["mse"]:.4f}\nSSIM: {dl_reconstr_metrics["ssim"]:.3f}\nWeights Rank: {rank_dl}\nProc. Err vs Shallow: {dl_sl_proc_error:.2e}'},
        {   'name': 'Deep Non-Linear AE',
            'latent': z_dnl.cpu().numpy(),
            'recon': rec_dnl.cpu().numpy(),
            "metrics":f'MSE: {dnl_reconstr_metrics["mse"]:.4f}\nSSIM: {dnl_reconstr_metrics["ssim"]:.3f}\nWeights Rank: {rank_dnl}'}
    ]
    plotVisualizer.plot_latent_space(models_data, imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names,save_path=f"{STATIC_ROOT}/slide_2")
    plotVisualizer.plot_sample_reconstructions(models_data, imgs_eval[:N_SAMPLES],save_path=f"{STATIC_ROOT}/slide_2")
    plt.show()
    plotVisualizer.plot_training_history([
        {'name': 'Shallow Linear AE', 'history': sl_loss_hist},
        {'name': 'Deep Linear AE', 'history': dl_loss_hist},
        {'name': 'Deep Non Linear AE', 'history': dnl_loss_hist},
    ])
    plt.show()


def run_slide_3():
    """Slide 3: AE vs VAE (Continuità Topologica)"""
    m_ae, ae_history = get_trained_model("deep_non_linear_ae", DeepAE,
                                      lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                                      latent_dim=LATENT_SPACE_DIM, non_linear=True)
    beta_vae_models = []
    beta_vae_training_history = []
    for beta in BETA_VALUES:
        beta_vae_model, history = get_trained_model(f"betavae_{beta}", VAE,
                              lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS,is_vae=True, beta=beta),
                              latent_dim=LATENT_SPACE_DIM)
        beta_vae_models.append(beta_vae_model)
        beta_vae_training_history.append(history)
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
    sil_scores = []
    ssim_scores = []
    mig_scores = []
    lower_beta_found, z_ae_np, proc_error = False, z_ae.cpu().numpy(), None
    for i, beta in enumerate(BETA_VALUES):
        if not lower_beta_found and beta <= 0.01:
            lower_beta_found = True
            z_ae_np, proc_error = latent_analizer.procrustes(z_ae_np, latent_list[i])
        mig_scores.append(latent_analizer.compute_mig(latent_list[i], labels_remapped))
        ssim_scores.append(latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_list[i])["ssim"])
        sil_scores.append(latent_analizer.compute_clustering_quality(latent_list[i], labels_remapped))

    models_data = [
        {   'name': f'Standard AE',
            'latent': z_ae_np,
            'recon': rec_ae.cpu().numpy() ,
            "metrics": f'SSIM: {ae_reconstr_metrics["ssim"]:.3f}\nMIG: {mig_ae:.3f}\nSilh: {sil_ae:.3f}' + (f"\nAligned with lowest beta VAE\nProc. Err={proc_error}" if lower_beta_found else '')},
    ]
    models_data.extend([
        {   'name': f'beta={beta} VAE',
            'latent': latent_list[i],
            'recon': rec_list[i],
            "metrics":f'SSIM={ssim_scores[i]:.3f}\nMIG: {mig_scores[i]:.3f}\nSilh: {sil_scores[i]:.3f}'}
        for i, beta in enumerate(BETA_VALUES)
    ])
    plotVisualizer.plot_latent_space(models_data, imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names,save_path=f"{STATIC_ROOT}/slide_3")
    plotVisualizer.plot_sample_reconstructions(models_data, imgs_eval[:N_SAMPLES],save_path=f"{STATIC_ROOT}/slide_3")
    plt.show()

    models_data = [
        {'name': 'Standard AE', 'history': ae_history}
    ]
    models_data.extend([
        {"name": f"beta={beta} VAE", "history": beta_vae_training_history[i]} for i, beta in enumerate(BETA_VALUES)
    ])
    plotVisualizer.plot_training_history(models_data)
    plt.show()


def run_slide_4():
    """Slide 4: ConvAE vs MLP VAE (Bias Induttivo e CCA)"""
    # choosing the middle beta value
    m_ae, ae_history = get_trained_model("deep_non_linear_ae", DeepAE,
                                         lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                                         latent_dim=LATENT_SPACE_DIM, non_linear=True)
    m_betavae, betavae_history = get_trained_model(f"betavae_{BETA_VALUES[1]}", VAE,
                              lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS,is_vae=True, beta=BETA_VALUES[1]),
                              latent_dim=LATENT_SPACE_DIM)
    m_conv, cnn_history = get_trained_model("conv_ae", ConvAE,
                               lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                               latent_dim=LATENT_SPACE_DIM)
    m_vit, vit_history = get_trained_model("transformer_ae", TransformerAE,
                                           lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                                           latent_dim=LATENT_SPACE_DIM)

    with torch.no_grad():
        rec_ae, z_ae = m_ae(imgs_eval.to(dm.device))
        rec_vae, mu_vae, _ = m_betavae(imgs_eval.to(dm.device))
        rec_conv, z_conv = m_conv(imgs_eval.to(dm.device))
        rec_vit, z_vit = m_vit(imgs_eval.to(dm.device))

    ssim_ae = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_ae)["ssim"]
    params_ae = latent_analizer.get_model_complexity(m_ae)
    silh_ae = latent_analizer.compute_clustering_quality(z_ae.cpu().numpy(), labels_remapped)
    flops_ae = latent_analizer.compute_flops(m_ae)["mflops"] # MegaFLOPs

    ssim_vae = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_vae)["ssim"]
    params_vae = latent_analizer.get_model_complexity(m_betavae)
    silh_vae = latent_analizer.compute_clustering_quality(mu_vae.cpu().numpy(), labels_remapped)
    flops_vae = latent_analizer.compute_flops(m_betavae)["mflops"] # MegaFLOPs

    # cnn metrics
    ssim_conv = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_conv)["ssim"]
    params_conv = latent_analizer.get_model_complexity(m_conv)
    silh_conv = latent_analizer.compute_clustering_quality(z_conv.cpu().numpy(), labels_remapped)
    flops_conv = latent_analizer.compute_flops(m_conv)["mflops"] # MegaFLOPs

    # vit metrics
    ssim_trans = latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_vit)["ssim"]
    params_trans = latent_analizer.get_model_complexity(m_vit)
    sil_trans = latent_analizer.compute_clustering_quality(z_vit.cpu().numpy(), labels_remapped)
    flops_vit = latent_analizer.compute_flops(m_vit)["mflops"]  # MegaFLOPs

    models_data = [
        {   'name': f'Standard AE',
            'latent': z_ae.cpu().numpy(),
            'recon': rec_ae.cpu().numpy(),
            "metrics":f'SSIM: {ssim_ae:.3f}\nSilh: {silh_ae:.3f}\nMegaFLOPs={flops_ae}\nParams count={params_ae:.2e}'},
        {   'name': f'beta={BETA_VALUES[1]} VAE',
            'latent': mu_vae.cpu().numpy(),
            'recon': rec_vae.cpu().numpy(),
            "metrics": f'SSIM: {ssim_vae:.3f}\nSilh: {silh_vae:.3f}\nMegaFLOPs={flops_vae}\nParams count={params_vae:.2e}'},
        {   'name': f'Conv AE',
            'latent': z_conv.cpu().numpy(),
            'recon': rec_conv.cpu().numpy(),
            "metrics": f'SSIM: {ssim_conv:.3f}\nSilh: {silh_conv:.3f}\nMegaFLOPs={flops_conv}\nParams count={params_conv:.2e}'},
        {
            'name': f'Transformer',
            'latent': z_vit.cpu().numpy(),
            'recon': rec_vit.cpu().numpy(),
            "metrics": f'SSIM: {ssim_trans:.3f}\nSilh: {sil_trans:.3f}\nMegaFLOPs={flops_vit}\nParams count={params_trans:.2e}'}
    ]
    plotVisualizer.plot_latent_space(models_data, imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names,save_path=f"{STATIC_ROOT}/slide_4")
    plotVisualizer.plot_sample_reconstructions(models_data, imgs_eval[:N_SAMPLES],save_path=f"{STATIC_ROOT}/slide_4")
    plt.show()
    plotVisualizer.plot_training_history([
        {"name": "Standard AE", "history": ae_history},
        {"name": f"beta={BETA_VALUES[1]} VAE", "history": betavae_history},
        {"name": "Conv AE", "history": cnn_history},
        {"name": "Transformer", "history": vit_history}
    ])
    plt.show()

