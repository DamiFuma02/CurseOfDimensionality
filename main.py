import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from core.LatentAnalizer import LatentAnalizer
from core.DataManager import DataManager
from core.models import LinearAE, DeepAE, VAE, ConvAE, TransformerAE
from core.ModelTrainer import ModelTrainer
from core.PlotVisualizer import PlotVisualizer
from core.constants import SEED, N_COMPONENTS_VIEW, LATENT_SPACE_DIM, EPOCHS, STATIC_ROOT, BATCH_SIZE, LIMIT_SAMPLES, \
    N_SAMPLES

# ==========================================
# GLOBAL CONFIGURATION & DETERMINISM
# ==========================================
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

dm = DataManager(batch_size=BATCH_SIZE, limit_samples=LIMIT_SAMPLES)
trainer = ModelTrainer(dm.device)
train_loader, eval_loader = dm.get_loaders()

imgs_eval, lbls_eval = next(iter(eval_loader))
labels_remapped = dm.remap_labels(lbls_eval)

MODELS_CACHE = {}


plotVisualizer = PlotVisualizer(n_components=N_COMPONENTS_VIEW)
latent_analizer = LatentAnalizer()

BETA_VALUES = [0.01,5,20]
assert len(BETA_VALUES) == 3


def get_trained_model(model_key, model_class, train_fn, **kwargs):
    """Utility updated to save the model and the loss history."""
    if model_key not in MODELS_CACHE:
        print(f"\n[Training] new model for: {model_key}")
        model = model_class(**kwargs)
        model, history = train_fn(model, train_loader, eval_loader)
        MODELS_CACHE[model_key] = {'model': model, 'history': history}
    return MODELS_CACHE[model_key]['model'], MODELS_CACHE[model_key]['history']

def get_trained_classifier(model_key, ae_model, is_vae=False):
    clf_key = f"clf_{model_key}"
    if clf_key not in MODELS_CACHE:
        print(f"[Training] Linear Classifier for: {model_key}")
        clf, history = trainer.train_classifier(
            ae_model, train_loader, eval_loader,
            latent_dim=LATENT_SPACE_DIM, epochs=EPOCHS, is_vae=is_vae
        )
        MODELS_CACHE[clf_key] = {'model': clf, 'history': history}
    return MODELS_CACHE[clf_key]['model'], MODELS_CACHE[clf_key]['history']

def get_predictions(ae_model, clf, is_vae=False):
    is_pca = not isinstance(ae_model, torch.nn.Module)
    if not is_pca:
        ae_model.eval()
    clf.eval()
    with torch.no_grad():
        if is_pca:
            x_flat = imgs_eval.view(imgs_eval.size(0),-1).numpy()
            z = torch.from_numpy(ae_model.transform(x_flat)).float().to(dm.device)
        elif is_vae:
            _, mu, sigma = ae_model(imgs_eval.to(dm.device))
            z = mu
        else:
            _, z, *_ = ae_model(imgs_eval.to(dm.device))
        logits = clf(z)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds

# ==========================================
# SLIDE SEQUENCE
# ==========================================

def run_slide_2():
    """Slide 2: Linearity vs. Depth Collapse (Rank and Non-Linearity)"""
    x_flat = imgs_eval.view(len(imgs_eval), -1).numpy()

    pca = PCA(n_components=LATENT_SPACE_DIM)
    z_pca = pca.fit_transform(x_flat - np.mean(x_flat, axis=0))
    rec_pca = pca.inverse_transform(z_pca) + np.mean(x_flat, axis=0)
    c_pca, pca_clf_hist = get_trained_classifier("pca", pca)
    pred_pca = get_predictions(pca, c_pca)

    m_sl, sl_loss_hist = get_trained_model("linear_ae", LinearAE,
                              lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                              latent_dim=LATENT_SPACE_DIM)
    c_sl, sl_clf_hist = get_trained_classifier("linear_ae", m_sl)
    pred_sl = get_predictions(m_sl, c_sl)

    m_dl, dl_loss_hist = get_trained_model("deep_linear_ae", DeepAE,
                                   lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                                   latent_dim=LATENT_SPACE_DIM, non_linear=False)
    c_dl, dl_clf_hist = get_trained_classifier("deep_linear_ae", m_dl)
    pred_dl = get_predictions(m_dl, c_dl)

    m_dnl, dnl_loss_hist = get_trained_model("deep_non_linear_ae", DeepAE,
                                      lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                                      latent_dim=LATENT_SPACE_DIM, non_linear=True)
    c_dnl, dnl_clf_hist = get_trained_classifier("deep_non_linear_ae", m_dnl)
    pred_dnl = get_predictions(m_dnl, c_dnl)


    with torch.no_grad():
        rec_sl, z_sl = m_sl(imgs_eval.to(dm.device))
        rec_dl, z_dl = m_dl(imgs_eval.to(dm.device))
        rec_dnl, z_dnl = m_dnl(imgs_eval.to(dm.device))

    rank_sl = latent_analizer.get_weight_rank(m_sl) # rank <= latent_dim,
    rank_dl = latent_analizer.get_weight_rank(m_dl) # rank <= latent_dim,
    # Procrustes between Shallow and Deep Linear [Networks] to show they represent the same space
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
            'predicted_labels': pred_pca,
            "metrics": f'MSE: {pca_reconstr_metrics["mse"]:.4f}\nSSIM: {pca_reconstr_metrics["ssim"]:.3g}\nProc. Err vs Shallow: {pca_sl_proc_error:.3g}\nAcc: {pca_clf_hist["val_acc"][-1]:.1f}%'},
        {   'name': 'Shallow Linear AE',
            'latent': z_sl.cpu().numpy(),
            'recon': rec_sl.cpu().numpy(),
            'predicted_labels': pred_sl,
            "metrics":f'MSE: {sl_reconstr_metrics["mse"]:.4f}\nSSIM: {sl_reconstr_metrics["ssim"]:.3g}\nWeights Rank: {rank_sl}\nAcc: {sl_clf_hist["val_acc"][-1]:.1f}%'},
        {   'name': 'Deep Linear AE',
            'latent': z_dl_sl_aligned,
            'recon': rec_dl.cpu().numpy(),
            'predicted_labels': pred_dl,
            "metrics": f'MSE: {dl_reconstr_metrics["mse"]:.4f}\nSSIM: {dl_reconstr_metrics["ssim"]:.3g}\nWeights Rank: {rank_dl}\nProc. Err vs Shallow: {dl_sl_proc_error:.3g}\nAcc: {dl_clf_hist["val_acc"][-1]:.1f}%'},
        {   'name': 'Deep Non-Linear AE',
            'latent': z_dnl.cpu().numpy(),
            'recon': rec_dnl.cpu().numpy(),
            'predicted_labels': pred_dnl,
            "metrics":f'MSE: {dnl_reconstr_metrics["mse"]:.4f}\nSSIM: {dnl_reconstr_metrics["ssim"]:.3g}\nAcc: {dnl_clf_hist["val_acc"][-1]:.1f}%'}
    ]
    plotVisualizer.plot_latent_space(models_data, imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names,save_path=f"{STATIC_ROOT}/slide_2")
    plotVisualizer.plot_sample_reconstructions(models_data, imgs_eval[:N_SAMPLES], labels_remapped[:N_SAMPLES], dm.semantic_names,save_path=f"{STATIC_ROOT}/slide_2")
    plt.show()
    plotVisualizer.plot_training_history([
        {'name': 'PCA', 'ae_history': None, 'clf_history': pca_clf_hist},
        {'name': 'Shallow Linear AE', 'ae_history': sl_loss_hist, 'clf_history': sl_clf_hist},
        {'name': 'Deep Linear AE', 'ae_history': dl_loss_hist, 'clf_history': dl_clf_hist},
        {'name': 'Deep Non Linear AE', 'ae_history': dnl_loss_hist, 'clf_history': dnl_clf_hist},
    ])
    plt.show()


def run_slide_3():
    """Slide 3: AE vs. VAE (Topological Continuity)"""
    m_ae, ae_history = get_trained_model("deep_non_linear_ae", DeepAE,
                                      lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                                      latent_dim=LATENT_SPACE_DIM, non_linear=True)
    c_ae, ae_clf_hist = get_trained_classifier("deep_non_linear_ae", m_ae)
    pred_ae = get_predictions(m_ae, c_ae)
    beta_vae_models = []
    beta_vae_training_history = []
    clf_beta_vae_training_histories = []
    clf_pred_vae_list = []
    acc_vae_list = []
    for beta in BETA_VALUES:
        beta_vae_model, history = get_trained_model(f"betavae_{beta}", VAE,
                              lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS,is_vae=True, beta=beta),
                              latent_dim=LATENT_SPACE_DIM)
        clf_vae, v_clf_hist = get_trained_classifier(f"betavae_{beta}", beta_vae_model, is_vae=True)
        pred_vae = get_predictions(beta_vae_model, clf_vae, is_vae=True)

        beta_vae_models.append(beta_vae_model)
        beta_vae_training_history.append(history)
        clf_beta_vae_training_histories.append(v_clf_hist)
        clf_pred_vae_list.append(pred_vae)
        acc_vae_list.append(v_clf_hist["val_acc"][-1])

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
    lowest_beta_found, z_ae_np, proc_error, = None, z_ae.cpu().numpy(), None
    for i, beta in enumerate(BETA_VALUES):
        if not lowest_beta_found and beta <= 0.01:
            lowest_beta_found = beta
            z_ae_np, proc_error = latent_analizer.procrustes(z_ae_np, latent_list[i])
        mig_scores.append(latent_analizer.compute_mig(latent_list[i], labels_remapped))
        ssim_scores.append(latent_analizer.compute_reconstruction_metrics(imgs_eval, rec_list[i])["ssim"])
        sil_scores.append(latent_analizer.compute_clustering_quality(latent_list[i], labels_remapped))

    models_data = [
        {   'name': f'Standard AE',
            'latent': z_ae_np,
            'recon': rec_ae.cpu().numpy() ,
            'predicted_labels': pred_ae,
            "metrics": f'SSIM: {ae_reconstr_metrics["ssim"]:.3g}\nMIG: {mig_ae:.3g}\nSilh: {sil_ae:.3g}\nAcc: {ae_clf_hist["val_acc"][-1]:.1f}%' + (f"\nProc. Err vs beta={lowest_beta_found} VAE: {proc_error:.3g}" if lowest_beta_found else '')},
    ]
    models_data.extend([
        {   'name': f'beta={beta} VAE',
            'latent': latent_list[i],
            'recon': rec_list[i],
            'predicted_labels': clf_pred_vae_list[i],
            "metrics":f'SSIM={ssim_scores[i]:.3g}\nMIG: {mig_scores[i]:.3g}\nSilh: {sil_scores[i]:.3g}\nAcc: {acc_vae_list[i]:.1f}%'}
        for i, beta in enumerate(BETA_VALUES)
    ])
    plotVisualizer.plot_latent_space(models_data, imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names,save_path=f"{STATIC_ROOT}/slide_3")
    plotVisualizer.plot_sample_reconstructions(models_data, imgs_eval[:N_SAMPLES], labels_remapped[:N_SAMPLES], dm.semantic_names,save_path=f"{STATIC_ROOT}/slide_3")
    plt.show()

    models_data = [
        {'name': 'Standard AE', 'ae_history': ae_history, 'clf_history': ae_clf_hist}
    ]
    models_data.extend([
        {"name": f"beta={beta} VAE", "ae_history": beta_vae_training_history[i], 'clf_history': clf_beta_vae_training_histories[i]} for i, beta in enumerate(BETA_VALUES)
    ])
    plotVisualizer.plot_training_history(models_data)
    plt.show()


def run_slide_4():
    """Slide 4: CNNAE vs VITAE"""

    m_conv, cnn_history = get_trained_model("conv_ae", ConvAE,
                               lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                               latent_dim=LATENT_SPACE_DIM)
    c_conv, conv_clf_hist = get_trained_classifier("conv_ae", m_conv)
    p_conv = get_predictions(m_conv, c_conv)

    m_vit, vit_history = get_trained_model("transformer_ae", TransformerAE,
                                           lambda m, tl, vl: trainer.train(m, tl, vl, epochs=EPOCHS),
                                           latent_dim=LATENT_SPACE_DIM)
    c_vit, vit_clf_hist = get_trained_classifier("transformer_ae", m_vit)
    p_vit = get_predictions(m_vit, c_vit)

    with torch.no_grad():
        rec_conv, z_conv, conv_enc_features = m_conv(imgs_eval.to(dm.device))
        rec_vit, z_vit, vit_enc_features = m_vit(imgs_eval.to(dm.device))

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
        {   'name': f'Conv AE',
            'latent': z_conv.cpu().numpy(),
            'recon': rec_conv.cpu().numpy(),
            'predicted_labels': p_conv,
            "metrics": f'SSIM: {ssim_conv:.3g}\nSilh: {silh_conv:.3g}\nMegaFLOPs={flops_conv:.3g}\nParams count={params_conv:.3g}\nAcc: {conv_clf_hist["val_acc"][-1]:.1f}%'},
        {
            'name': f'Transformer',
            'latent': z_vit.cpu().numpy(),
            'recon': rec_vit.cpu().numpy(),
            'predicted_labels': p_vit,
            "metrics": f'SSIM: {ssim_trans:.3g}\nSilh: {sil_trans:.3g}\nMegaFLOPs={flops_vit:.3g}\nParams count={params_trans:.3g}\nAcc: {vit_clf_hist["val_acc"][-1]:.1f}%'}
    ]
    plotVisualizer.plot_latent_space(models_data, imgs_eval[:N_SAMPLES], labels_remapped, dm.semantic_names,save_path=f"{STATIC_ROOT}/slide_4")
    plotVisualizer.plot_sample_reconstructions(models_data, imgs_eval[:N_SAMPLES], labels_remapped[:N_SAMPLES], dm.semantic_names,save_path=f"{STATIC_ROOT}/slide_4")
    plt.show()
    plotVisualizer.plot_training_history([
        {"name": "Conv AE", "ae_history": cnn_history, 'clf_history': conv_clf_hist},
        {"name": "Transformer", "ae_history": vit_history, 'clf_history': vit_clf_hist}
    ])
    plt.show()

    plotVisualizer.plot_cka_heatmap(conv_enc_features,vit_enc_features,save_path=f"{STATIC_ROOT}/slide_4")
    plt.show()

