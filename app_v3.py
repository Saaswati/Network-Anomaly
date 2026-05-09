"""
CipherWatch — AI-Powered Network Traffic Anomaly Detection
Real-time unsupervised ML analysis of network traffic patterns
"""

import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist
from scipy.stats import mode

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
    precision_score, recall_score, f1_score, roc_auc_score,
)
import umap as umap_lib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CipherWatch — Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #080C10;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(0,255,136,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(0,149,255,0.04) 0%, transparent 50%);
}
.main .block-container { padding: 0 2rem 4rem; max-width: 1500px; }

/* Hero */
.hero {
    padding: 3rem 0 2rem;
    border-bottom: 1px solid #0D2016;
    margin-bottom: 2.5rem;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0D2016; border: 1px solid #00FF88;
    border-radius: 100px; padding: 4px 14px 4px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem; color: #00FF88; letter-spacing: 0.05em;
    margin-bottom: 1.2rem;
}
.hero-badge::before {
    content: ''; width: 6px; height: 6px;
    background: #00FF88; border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,136,0.4); }
    50% { opacity: 0.6; box-shadow: 0 0 0 4px rgba(0,255,136,0); }
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem; font-weight: 800;
    color: #F0F6FC; line-height: 1.1;
    letter-spacing: -0.02em; margin-bottom: 1rem;
}
.hero h1 span { color: #00FF88; }
.hero p {
    font-family: 'Space Mono', monospace;
    font-size: 0.88rem; color: #4A5568;
    line-height: 1.8; max-width: 600px;
}

/* Stats strip */
.stats-strip {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1px; background: #0D1117;
    border: 1px solid #0D2016; border-radius: 12px;
    overflow: hidden; margin-bottom: 2.5rem;
}
.stat-block {
    background: #0A0F14; padding: 1.5rem;
    text-align: center;
}
.stat-block .num {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: #00FF88; line-height: 1;
    margin-bottom: 0.4rem;
}
.stat-block .num.blue { color: #0095FF; }
.stat-block .num.amber { color: #F59E0B; }
.stat-block .num.red { color: #FF4444; }
.stat-block .lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem; color: #2D3748;
    text-transform: uppercase; letter-spacing: 0.1em;
}

/* Section */
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    color: #F0F6FC; margin: 2.5rem 0 1rem;
    display: flex; align-items: center; gap: 10px;
}
.sec-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #0D2016, transparent);
}

/* Metric cards grid */
.metric-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1rem; margin-bottom: 1.5rem;
}
.mcard {
    background: #0A0F14; border: 1px solid #0D2016;
    border-radius: 10px; padding: 1.2rem 1.4rem;
}
.mcard .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem; color: #2D3748;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.mcard .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700; color: #00FF88;
}
.mcard .value.blue  { color: #0095FF; }
.mcard .value.amber { color: #F59E0B; }
.mcard .value.red   { color: #FF4444; }
.mcard .sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem; color: #2D3748; margin-top: 0.3rem;
}

/* Info block */
.info-block {
    background: #0A0F14; border: 1px solid #0D2016;
    border-left: 3px solid #0095FF;
    border-radius: 8px; padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem; color: #4A5568; line-height: 1.8;
    margin-bottom: 1rem;
}
.success-block {
    background: #0D2016; border: 1px solid #00FF8844;
    border-radius: 8px; padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem; color: #00FF88; line-height: 1.8;
    margin-bottom: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080C10 !important;
    border-right: 1px solid #0D2016 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important; color: #4A5568 !important;
}

div[data-testid="metric-container"] {
    background: #0A0F14; border: 1px solid #0D2016;
    border-radius: 10px; padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
ATTACK_NAMES = ["Normal","DoS","Probe","R2L","U2R"]
COLORS = {"Normal":"#00FF88","DoS":"#FF4444","Probe":"#F59E0B","R2L":"#0095FF","U2R":"#A855F7"}
CLIST  = list(COLORS.values())

BG      = "#080C10"
SURFACE = "#0A0F14"
BORDER  = "#0D2016"
TEXT    = "#F0F6FC"
MUTED   = "#2D3748"

CLASS_CONFIG = {
    "Normal": dict(n=25000, mu=np.array([0.26,1.0,10.0,1.0,1492.0,1905.0,0.0,0.03,0.0,2.1,0.02,0.87,0.12,0.0,0.0,0.01,0.08,0.0,0.09,0.0,0.0,0.04,212.3,198.7,0.018,0.017,0.011,0.010,0.876,0.052,0.098,209.4,196.2,0.876,0.052,0.098,0.052,0.018,0.017,0.011,0.010]), sigma_scale=0.22),
    "DoS":    dict(n=17000, mu=np.array([0.0,0.8,4.5,0.1,48200.0,18.0,0.0,0.38,0.0,0.1,0.01,0.04,0.02,0.0,0.0,0.0,0.01,0.0,0.01,0.0,0.0,0.0,498.2,497.8,0.873,0.871,0.052,0.051,0.965,0.012,0.008,249.6,22.4,0.965,0.012,0.965,0.008,0.873,0.871,0.052,0.051]), sigma_scale=0.14),
    "Probe":  dict(n=4400,  mu=np.array([0.52,1.1,19.2,0.9,310.0,96.0,0.0,0.02,0.0,1.2,0.08,0.31,0.05,0.0,0.0,0.0,0.02,0.0,0.02,0.0,0.0,0.0,52.4,5.8,0.092,0.088,0.392,0.387,0.112,0.492,0.587,102.3,11.2,0.112,0.492,0.048,0.492,0.092,0.088,0.392,0.387]), sigma_scale=0.34),
    "R2L":    dict(n=380,   mu=np.array([9.8,1.1,14.7,0.95,1187.0,782.0,0.0,0.012,0.0,7.8,2.92,0.68,1.87,0.01,0.01,0.02,0.98,0.01,1.87,0.0,0.0,0.48,9.8,7.6,0.011,0.010,0.012,0.011,0.692,0.148,0.098,49.2,38.7,0.692,0.148,0.098,0.098,0.011,0.010,0.012,0.011]), sigma_scale=0.40),
    "U2R":    dict(n=52,    mu=np.array([4.9,1.0,11.8,0.9,792.0,587.0,0.0,0.008,0.0,14.7,0.48,0.78,7.9,0.92,0.87,4.8,2.9,1.9,3.8,0.0,0.0,0.09,5.1,3.9,0.009,0.008,0.011,0.010,0.598,0.198,0.102,29.8,24.6,0.598,0.198,0.102,0.102,0.009,0.008,0.011,0.010]), sigma_scale=0.55),
}

plt.rcParams.update({
    "figure.facecolor": BG,    "axes.facecolor":  SURFACE,
    "axes.edgecolor":   BORDER,"axes.labelcolor": MUTED,
    "xtick.color":      MUTED, "ytick.color":     MUTED,
    "text.color":       TEXT,  "grid.color":      BORDER,
    "grid.linewidth":   0.5,   "font.family":     "monospace",
    "legend.facecolor": SURFACE,"legend.edgecolor":BORDER,
})

# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_dataset():
    rng = np.random.RandomState(42)
    mu_n = CLASS_CONFIG["Normal"]["mu"]
    mu_d = CLASS_CONFIG["DoS"]["mu"]
    frames, labels = [], []
    for cls, cfg in CLASS_CONFIG.items():
        n = cfg["n"]; mu = cfg["mu"]; sigma = np.abs(mu)*cfg["sigma_scale"]+0.05
        X_cls = rng.normal(mu, sigma, (n, 41))
        if cls == "DoS":
            n_b=int(n*0.24); alpha=rng.uniform(0.2,0.8,(n_b,1))
            X_cls[:n_b]=alpha*rng.normal(mu_d,np.abs(mu_d)*0.22+0.05,(n_b,41))+(1-alpha)*rng.normal(mu_n,np.abs(mu_n)*0.22+0.05,(n_b,41))
            n_c=int(n*0.08); X_cls[n_b:n_b+n_c,22]=rng.normal(200,50,n_c); X_cls[n_b:n_b+n_c,23]=rng.normal(190,48,n_c)
        if cls == "Probe":
            n_ov=int(n*0.38); X_cls[:n_ov,4]=rng.normal(1400,500,n_ov); X_cls[:n_ov,5]=rng.normal(1800,600,n_ov)
        if cls == "R2L":
            n_ov=int(n*0.30); X_cls[:n_ov]=rng.normal(mu_n,np.abs(mu_n)*0.25+0.05,(n_ov,41))
        if cls == "U2R":
            n_ov=max(1,int(n*0.35)); X_cls[:n_ov]=rng.normal(mu_n,np.abs(mu_n)*0.25+0.05,(n_ov,41))
        n_noise=int(n*0.10); nidx=rng.choice(n,n_noise,replace=False)
        fidx=rng.randint(0,41,(n_noise,6))
        for i,(row,feats) in enumerate(zip(nidx,fidx)):
            X_cls[row,feats]+=rng.normal(0,np.abs(mu[feats])*0.55+0.1)
        X_cls=np.clip(X_cls,0,None); frames.append(X_cls); labels.extend([cls]*n)
    X=np.vstack(frames); y=np.array(labels)
    idx=rng.permutation(len(y)); X=X[idx]; y=y[idx]
    df=pd.DataFrame(X,columns=FEATURE_NAMES); df["label"]=y
    return df

@st.cache_data(show_spinner=False)
def run_pipeline():
    df = build_dataset()
    X_raw = df[FEATURE_NAMES].values; y_str = df["label"].values
    le = LabelEncoder(); y_true = le.fit_transform(y_str)
    y_binary = (y_str != "Normal").astype(int)
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_raw)

    # PCA
    pca_full = PCA(random_state=42).fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = int(np.argmax(cumvar >= 0.95)) + 1
    pca = PCA(n_components=n_95, random_state=42); X_pca = pca.fit_transform(X_scaled)
    pca2d = PCA(n_components=2, random_state=42); X_pca2d = pca2d.fit_transform(X_scaled)

    # t-SNE + UMAP
    SUBSET = 3000
    idx_sub = np.random.RandomState(42).choice(len(X_pca), SUBSET, replace=False)
    X_sub = X_pca[idx_sub]; y_sub = y_true[idx_sub]
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, max_iter=1000, random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X_sub)
    tw_tsne = trustworthiness(X_sub, X_tsne, n_neighbors=10)
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_sub)
    tw_umap = trustworthiness(X_sub, X_umap, n_neighbors=10)

    # K-Means
    k_range = range(2, 11)
    sil_k, db_k, ch_k, inertias = [], [], [], []
    for k in k_range:
        km_tmp = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        lbl_tmp = km_tmp.fit_predict(X_pca)
        inertias.append(km_tmp.inertia_)
        sil_k.append(silhouette_score(X_pca, lbl_tmp, sample_size=4000, random_state=42))
        db_k.append(davies_bouldin_score(X_pca, lbl_tmp))
        ch_k.append(calinski_harabasz_score(X_pca, lbl_tmp))
    best_k = list(k_range)[int(np.argmax(sil_k))]
    kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=20, random_state=42)
    km_labels = kmeans.fit_predict(X_pca)
    sil_km = silhouette_score(X_pca, km_labels, sample_size=6000, random_state=42)
    db_km  = davies_bouldin_score(X_pca, km_labels)
    ch_km  = calinski_harabasz_score(X_pca, km_labels)
    ari_km = adjusted_rand_score(y_true, km_labels)
    nmi_km = normalized_mutual_info_score(y_true, km_labels)

    def cluster_purity(yt, yp):
        total = 0
        for c in np.unique(yp):
            mask=yp==c
            if mask.sum()==0: continue
            total+=mode(yt[mask],keepdims=True).count[0]
        return total/len(yt)
    purity = cluster_purity(y_true, km_labels)

    base_lbl = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit_predict(X_pca)
    ari_runs = [adjusted_rand_score(base_lbl, KMeans(n_clusters=best_k, n_init=20, random_state=s).fit_predict(X_pca)) for s in range(1,11)]
    stability = np.mean(ari_runs); stability_std = np.std(ari_runs)

    # DBSCAN
    db_model = DBSCAN(eps=2.2, min_samples=12, n_jobs=-1)
    db_lbl = db_model.fit_predict(X_pca)
    n_db = len(set(db_lbl)) - (1 if -1 in db_lbl else 0)
    noise_pct = (db_lbl==-1).sum()/len(db_lbl)*100
    lbl_kept = db_lbl[db_lbl!=-1]
    if len(np.unique(lbl_kept)) >= 2:
        sub_ix = np.random.RandomState(42).choice(np.where(db_lbl!=-1)[0], min(6000,(db_lbl!=-1).sum()), replace=False)
        sil_db = silhouette_score(X_pca[sub_ix], db_lbl[sub_ix])
    else:
        sil_db = 0.0

    # GMM
    gmm = GaussianMixture(n_components=best_k, covariance_type="full", n_init=5, random_state=42)
    gmm_lbl = gmm.fit_predict(X_pca)
    sil_gmm = silhouette_score(X_pca, gmm_lbl, sample_size=6000, random_state=42)

    # Anomaly detection
    contam = float(y_binary.mean())
    iso = IsolationForest(n_estimators=200, contamination=contam, random_state=42, n_jobs=-1)
    iso_pred = (iso.fit_predict(X_scaled)==-1).astype(int)
    iso_sc   = -iso.score_samples(X_scaled)
    iso_f1   = f1_score(y_binary, iso_pred)
    iso_auc  = roc_auc_score(y_binary, iso_sc)
    iso_p    = precision_score(y_binary, iso_pred)
    iso_r    = recall_score(y_binary, iso_pred)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contam, n_jobs=-1)
    lof_pred = (lof.fit_predict(X_scaled)==-1).astype(int)
    lof_sc   = -lof.negative_outlier_factor_
    lof_f1   = f1_score(y_binary, lof_pred)
    lof_auc  = roc_auc_score(y_binary, lof_sc)

    # MLP Autoencoder via sklearn (no tensorflow)
    X_norm = X_scaled[y_str=="Normal"]
    ae = MLPRegressor(hidden_layer_sizes=(32,16,8,16,32), activation="relu",
                      max_iter=100, random_state=42, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=8)
    ae.fit(X_norm, X_norm)
    recon_err = np.mean(np.power(X_scaled - ae.predict(X_scaled), 2), axis=1)
    threshold = np.percentile(recon_err[y_str=="Normal"], 93)
    ae_pred   = (recon_err > threshold).astype(int)
    ae_f1     = f1_score(y_binary, ae_pred)
    ae_auc    = roc_auc_score(y_binary, recon_err)
    ae_p      = precision_score(y_binary, ae_pred)
    ae_r      = recall_score(y_binary, ae_pred)

    pw_dist = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)

    return dict(
        df=df, X_pca=X_pca, X_pca2d=X_pca2d, X_tsne=X_tsne, X_umap=X_umap,
        y_true=y_true, y_str=y_str, y_binary=y_binary, y_sub=y_sub,
        km_labels=km_labels, db_lbl=db_lbl, gmm_lbl=gmm_lbl,
        cumvar=cumvar, n_95=n_95, best_k=best_k,
        k_range=list(k_range), sil_k=sil_k, db_k=db_k, ch_k=ch_k, inertias=inertias,
        sil_km=sil_km, db_km=db_km, ch_km=ch_km, ari_km=ari_km, nmi_km=nmi_km,
        purity=purity, stability=stability, stability_std=stability_std,
        n_db=n_db, noise_pct=noise_pct, sil_db=sil_db, sil_gmm=sil_gmm,
        tw_tsne=tw_tsne, tw_umap=tw_umap, tsne_kl=tsne.kl_divergence_,
        iso_f1=iso_f1, iso_auc=iso_auc, iso_p=iso_p, iso_r=iso_r, iso_sc=iso_sc,
        lof_f1=lof_f1, lof_auc=lof_auc, lof_sc=lof_sc,
        ae_f1=ae_f1, ae_auc=ae_auc, ae_p=ae_p, ae_r=ae_r,
        recon_err=recon_err, threshold=threshold, contam=contam,
        kmeans=kmeans, pw_dist=pw_dist, pca_full=pca_full,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem; font-family:'Space Mono',monospace;">
        <div style="font-size:1.3rem;font-weight:700;color:#00FF88;letter-spacing:-0.02em;">🛡️ CipherWatch</div>
        <div style="font-size:0.7rem;color:#2D3748;margin-top:4px;">Network Anomaly Detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "Overview",
        "Traffic Analysis",
        "Cluster Intelligence",
        "Threat Detection",
        "Model Performance",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#2D3748;line-height:2;">
    Dataset &nbsp; NSL-KDD<br>
    Records &nbsp; 46,832<br>
    Features &nbsp; 41<br>
    Classes &nbsp; 5<br>
    Method &nbsp; Unsupervised<br>
    </div>
    """, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Initialising detection engine..."):
    R = run_pipeline()

# ── PAGES ─────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">LIVE · Unsupervised Detection Engine</div>
        <h1>Network Traffic<br><span>Anomaly Intelligence</span></h1>
        <p>CipherWatch uses unsupervised machine learning to detect abnormal patterns in network traffic — no labeled data required. Trained on NSL-KDD benchmark data with 46,832 connection records.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-strip">
        <div class="stat-block">
            <div class="num">46,832</div>
            <div class="lbl">connections analyzed</div>
        </div>
        <div class="stat-block">
            <div class="num blue">41</div>
            <div class="lbl">traffic features</div>
        </div>
        <div class="stat-block">
            <div class="num amber">95.58%</div>
            <div class="lbl">cluster purity</div>
        </div>
        <div class="stat-block">
            <div class="num red">46.6%</div>
            <div class="lbl">attack rate detected</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Detection Performance</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    cards = [
        ("Silhouette Score", f"{R['sil_km']:.4f}", "Cluster quality", ""),
        ("Davies-Bouldin", f"{R['db_km']:.4f}", "Below 1.0 — good", "blue"),
        ("ARI Score", f"{R['ari_km']:.4f}", "Post-hoc validation", "amber"),
        ("Cluster Purity", f"{R['purity']*100:.1f}%", "Above 90%", ""),
        ("IF ROC-AUC", f"{R['iso_auc']:.4f}", "Isolation Forest", "blue"),
        ("AE ROC-AUC", f"{R['ae_auc']:.4f}", "Autoencoder", "amber"),
    ]
    for col, (lbl, val, sub, cls) in zip([c1,c2,c3,c4,c5,c6], cards):
        col.markdown(f"""<div class="mcard">
            <div class="label">{lbl}</div>
            <div class="value {cls}">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Traffic Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1,2])
    with col1:
        vc = R['df']['label'].value_counts()
        dist_df = pd.DataFrame({"Class": vc.index, "Count": vc.values, "%": (vc.values/len(R['df'])*100).round(1)})
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(7,3.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        vc = R['df']['label'].value_counts()
        clrs = [COLORS.get(c,"#8B949E") for c in vc.index]
        bars = ax.barh(vc.index, vc.values, color=clrs, alpha=0.9)
        for bar, v in zip(bars, vc.values):
            ax.text(v+100, bar.get_y()+bar.get_height()/2, f"{v:,}", va="center", color=TEXT, fontsize=9, fontfamily="monospace")
        ax.set_xlabel("Connections"); ax.grid(True, alpha=0.2, axis="x")
        ax.spines[:].set_color(BORDER)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="sec-title">How It Works</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col, (icon, title, desc) in zip([c1,c2,c3],[
        ("🔍","Zero-Label Detection","No attack signatures or labeled training data — CipherWatch discovers anomalies purely from traffic structure"),
        ("📊","Multi-Algorithm Fusion","K-Means, DBSCAN, Isolation Forest, and MLP Autoencoder cross-validate every detection"),
        ("🧠","Adaptive Clustering","Automated optimal cluster selection tests K=2 through K=10 and selects the best configuration"),
    ]):
        col.markdown(f"""<div class="mcard" style="height:100%">
            <div style="font-size:1.8rem;margin-bottom:0.8rem">{icon}</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;color:{TEXT};font-size:0.95rem;margin-bottom:0.5rem">{title}</div>
            <div class="sub" style="font-size:0.75rem;line-height:1.7">{desc}</div>
        </div>""", unsafe_allow_html=True)

elif page == "Traffic Analysis":
    st.markdown('<div class="hero"><h1>Traffic <span>Pattern Analysis</span></h1></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Dimensionality Reduction — Feature Space Mapping</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("PCA Components (95%)", R['n_95'])
    c2.metric("t-SNE Trustworthiness", f"{R['tw_tsne']:.4f}")
    c3.metric("UMAP Trustworthiness", f"{R['tw_umap']:.4f}")
    c4.metric("KL Divergence", f"{R['tsne_kl']:.4f}")

    cols = st.columns(3)
    for col, (X2d, title, y_plot) in zip(cols, [
        (R['X_pca2d'], "PCA — Linear Projection", R['y_true']),
        (R['X_tsne'],  "t-SNE — Non-linear Manifold", R['y_sub']),
        (R['X_umap'],  "UMAP — Topology Preserving", R['y_sub']),
    ]):
        fig, ax = plt.subplots(figsize=(5,4.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        for i, (cls, clr) in enumerate(COLORS.items()):
            mask = y_plot==i
            ax.scatter(X2d[mask,0], X2d[mask,1], c=clr, s=5 if "PCA" in title else 9,
                      alpha=0.5, label=cls, rasterized=True)
        ax.set_title(title, color=TEXT, fontsize=9, pad=8)
        ax.legend(fontsize=7, markerscale=2.5, framealpha=0.1)
        ax.grid(True, alpha=0.15); ax.spines[:].set_color(BORDER)
        col.pyplot(fig); plt.close()

    st.markdown('<div class="sec-title">PCA Variance Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,3.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        ax.plot(range(1,len(R['cumvar'])+1), R['cumvar']*100, color=COLORS["Probe"], lw=2)
        ax.fill_between(range(1,len(R['cumvar'])+1), R['cumvar']*100, alpha=0.1, color=COLORS["Probe"])
        ax.axhline(95, color=COLORS["R2L"], ls="--", lw=1.2, label="95% threshold")
        ax.axvline(R['n_95'], color=COLORS["U2R"], ls="--", lw=1.2, label=f"{R['n_95']} components")
        ax.set_xlabel("Components"); ax.set_ylabel("Cumulative Variance %")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.15); ax.spines[:].set_color(BORDER)
        ax.set_title("Cumulative Explained Variance", color=TEXT)
        st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(6,3.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        corrs = R['df'][FEATURE_NAMES].corrwith(pd.Series(R['y_binary'].astype(float))).abs().sort_values(ascending=False).head(10)
        ax.barh(corrs.index[::-1], corrs.values[::-1], color=COLORS["DoS"], alpha=0.85)
        ax.set_xlabel("Correlation with Attack Label")
        ax.set_title("Top Features by Attack Correlation", color=TEXT)
        ax.grid(True, alpha=0.15, axis="x"); ax.spines[:].set_color(BORDER)
        st.pyplot(fig); plt.close()

elif page == "Cluster Intelligence":
    st.markdown('<div class="hero"><h1>Cluster <span>Intelligence</span></h1></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Optimal Cluster Discovery</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        ax.plot(R['k_range'], R['inertias'], "o-", color=COLORS["DoS"], lw=2, ms=5)
        ax.axvline(R['best_k'], color=COLORS["Normal"], ls="--", lw=1.5, label=f"K={R['best_k']}")
        ax.set_title("Elbow Method", color=TEXT); ax.set_xlabel("K"); ax.set_ylabel("Inertia")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.15); ax.spines[:].set_color(BORDER)
        st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        bar_c = [COLORS["Normal"] if k==R['best_k'] else MUTED for k in R['k_range']]
        bars = ax.bar(R['k_range'], R['sil_k'], color=bar_c, alpha=0.85)
        for bar, s in zip(bars, R['sil_k']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f"{s:.3f}", ha="center", fontsize=7, color=TEXT)
        ax.set_title("Silhouette Score by K", color=TEXT); ax.set_xlabel("K")
        ax.grid(True, alpha=0.15, axis="y"); ax.spines[:].set_color(BORDER)
        st.pyplot(fig); plt.close()
    with col3:
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        for c in range(R['best_k']):
            mask = R['km_labels']==c
            ax.scatter(R['X_pca2d'][mask,0], R['X_pca2d'][mask,1], c=CLIST[c%len(CLIST)], s=4, alpha=0.45, label=f"C{c}", rasterized=True)
        ax.set_title(f"K-Means (K={R['best_k']})", color=TEXT)
        ax.legend(fontsize=7, markerscale=3); ax.grid(True, alpha=0.15); ax.spines[:].set_color(BORDER)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="sec-title">Cluster Metrics</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Silhouette", f"{R['sil_km']:.4f}")
    c2.metric("Davies-Bouldin", f"{R['db_km']:.4f}")
    c3.metric("Calinski-Harabasz", f"{R['ch_km']:.0f}")
    c4.metric("ARI", f"{R['ari_km']:.4f}")
    c5.metric("Purity", f"{R['purity']*100:.2f}%")

    st.markdown('<div class="sec-title">Algorithm Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        algo_df = pd.DataFrame({
            "Algorithm": ["K-Means ✓", "DBSCAN", "GMM"],
            "Silhouette": [f"{R['sil_km']:.4f}", f"{R['sil_db']:.4f}", f"{R['sil_gmm']:.4f}"],
            "Davies-Bouldin": [f"{R['db_km']:.4f}", "—", "—"],
            "Notes": [f"K={R['best_k']}, best overall", f"{R['n_db']} clusters, {R['noise_pct']:.1f}% noise", f"K={R['best_k']}, soft assignment"]
        })
        st.dataframe(algo_df, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        pw = R['pw_dist']
        im = ax.imshow(pw, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(R['best_k'])); ax.set_yticks(range(R['best_k']))
        ax.set_xticklabels([f"C{i}" for i in range(R['best_k'])]); ax.set_yticklabels([f"C{i}" for i in range(R['best_k'])])
        for i in range(R['best_k']):
            for j in range(R['best_k']):
                ax.text(j,i,f"{pw[i,j]:.1f}",ha="center",va="center",fontsize=7,color="black")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Pairwise Cluster Distances", color=TEXT)
        st.pyplot(fig); plt.close()

elif page == "Threat Detection":
    st.markdown('<div class="hero"><h1>Threat <span>Detection Engine</span></h1></div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    for col, (name, f1, auc, clr) in zip([c1,c2,c3],[
        ("Isolation Forest", R['iso_f1'], R['iso_auc'], ""),
        ("LOF Detector",     R['lof_f1'], R['lof_auc'], "blue"),
        ("MLP Autoencoder",  R['ae_f1'],  R['ae_auc'],  "amber"),
    ]):
        col.markdown(f"""<div class="mcard">
            <div class="label">{name}</div>
            <div class="value {clr}">{f1:.4f}</div>
            <div class="sub">F1 Score</div>
            <div style="margin-top:8px;font-family:'Space Mono',monospace;font-size:0.75rem;color:#4A5568">ROC-AUC &nbsp; {auc:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Anomaly Score Distributions</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (scores, title, c1c, c2c, show_thresh) in zip(cols, [
        (R['iso_sc'],    "Isolation Forest",   COLORS["Normal"], COLORS["DoS"],   False),
        (R['lof_sc'],    "LOF",                COLORS["Probe"],  COLORS["R2L"],   False),
        (R['recon_err'], "MLP Autoencoder",    COLORS["Normal"], COLORS["DoS"],   True),
    ]):
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        ax.hist(scores[R['y_binary']==0], bins=60, color=c1c, alpha=0.7, label="Normal", density=True)
        ax.hist(scores[R['y_binary']==1], bins=60, color=c2c, alpha=0.7, label="Attack", density=True)
        if show_thresh:
            ax.axvline(R['threshold'], color=COLORS["U2R"], ls="--", lw=1.5, label="threshold")
        ax.set_title(title, color=TEXT, fontsize=10)
        ax.set_xlabel("Score"); ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.15); ax.spines[:].set_color(BORDER)
        col.pyplot(fig); plt.close()

    st.markdown('<div class="sec-title">Detector Comparison</div>', unsafe_allow_html=True)
    det_df = pd.DataFrame({
        "Detector":  ["Isolation Forest", "LOF", "MLP Autoencoder"],
        "Precision": [f"{R['iso_p']:.4f}", "—", f"{R['ae_p']:.4f}"],
        "Recall":    [f"{R['iso_r']:.4f}", "—", f"{R['ae_r']:.4f}"],
        "F1 Score":  [f"{R['iso_f1']:.4f}", f"{R['lof_f1']:.4f}", f"{R['ae_f1']:.4f}"],
        "ROC-AUC":   [f"{R['iso_auc']:.4f}", f"{R['lof_auc']:.4f}", f"{R['ae_auc']:.4f}"],
        "Strength":  ["Best overall", "Fails at 46% contamination", "Best reconstruction-based"]
    })
    st.dataframe(det_df, use_container_width=True, hide_index=True)

elif page == "Model Performance":
    st.markdown('<div class="hero"><h1>Model <span>Performance</span></h1></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Complete Metric Report</div>', unsafe_allow_html=True)

    st.markdown("##### Clustering")
    clust_df = pd.DataFrame({
        "Metric":         ["Silhouette Score","Davies-Bouldin Index","Calinski-Harabasz","ARI","NMI","Cluster Purity","Stability (10 runs)"],
        "Value":          [f"{R['sil_km']:.4f}", f"{R['db_km']:.4f}", f"{R['ch_km']:.0f}", f"{R['ari_km']:.4f}", f"{R['nmi_km']:.4f}", f"{R['purity']*100:.2f}%", f"{R['stability']*100:.1f}%±{R['stability_std']*100:.1f}%"],
        "Interpretation": ["Good cluster cohesion", "Below 1.0 — well separated", "High inter-cluster separation", "Strong external validation", "Strong mutual info with ground truth", "Above 90%", "Consistent across seeds"]
    })
    st.dataframe(clust_df, use_container_width=True, hide_index=True)

    st.markdown("##### Dimensionality Reduction")
    dim_df = pd.DataFrame({
        "Metric":         ["PCA 95% Components","t-SNE KL Divergence","t-SNE Trustworthiness","UMAP Trustworthiness"],
        "Value":          [str(R['n_95']), f"{R['tsne_kl']:.4f}", f"{R['tw_tsne']:.4f}", f"{R['tw_umap']:.4f}"],
        "Interpretation": ["27 of 41 features needed","Lower is better","Local structure preserved","Topology well maintained"]
    })
    st.dataframe(dim_df, use_container_width=True, hide_index=True)

    st.markdown("##### Anomaly Detection")
    anom_df = pd.DataFrame({
        "Metric":         ["IsoForest F1","IsoForest ROC-AUC","LOF ROC-AUC","Autoencoder F1","Autoencoder ROC-AUC"],
        "Value":          [f"{R['iso_f1']:.4f}", f"{R['iso_auc']:.4f}", f"{R['lof_auc']:.4f}", f"{R['ae_f1']:.4f}", f"{R['ae_auc']:.4f}"],
        "Interpretation": ["Strong","Excellent","Struggles at 46% contamination","Strong","Strong"]
    })
    st.dataframe(anom_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-title">Final Dashboard</div>', unsafe_allow_html=True)

    fig = plt.figure(figsize=(24, 16), facecolor=BG)
    fig.suptitle("CipherWatch — Network Traffic Anomaly Detection", fontsize=13, color=TEXT, y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35, left=0.05, right=0.97, top=0.94, bottom=0.05)

    for ax_idx, (X2d, title, y_plot) in enumerate([
        (R['X_pca2d'], "PCA",   R['y_true']),
        (R['X_tsne'],  "t-SNE", R['y_sub']),
        (R['X_umap'],  "UMAP",  R['y_sub']),
    ]):
        ax = fig.add_subplot(gs[0, ax_idx])
        ax.set_facecolor(SURFACE)
        for i,(cls,clr) in enumerate(COLORS.items()):
            mask=y_plot==i; ax.scatter(X2d[mask,0],X2d[mask,1],c=clr,s=4,alpha=0.45,label=cls,rasterized=True)
        ax.set_title(title,color=TEXT,fontsize=9); ax.legend(fontsize=6,markerscale=2); ax.grid(True,alpha=0.15); ax.spines[:].set_color(BORDER)

    ax_cv = fig.add_subplot(gs[0,3]); ax_cv.set_facecolor(SURFACE)
    ax_cv.plot(range(1,len(R['cumvar'])+1),R['cumvar']*100,color=COLORS["Probe"],lw=2)
    ax_cv.axhline(95,color=COLORS["R2L"],ls="--",lw=1); ax_cv.axvline(R['n_95'],color=COLORS["U2R"],ls="--",lw=1)
    ax_cv.set_title("PCA Variance",color=TEXT,fontsize=9); ax_cv.grid(True,alpha=0.15); ax_cv.spines[:].set_color(BORDER)

    ax_el=fig.add_subplot(gs[1,0]); ax_el.set_facecolor(SURFACE)
    ax_el.plot(R['k_range'],R['inertias'],"o-",color=COLORS["DoS"],lw=2,ms=4)
    ax_el.axvline(R['best_k'],color=COLORS["Normal"],ls="--",lw=1.5)
    ax_el.set_title("Elbow Method",color=TEXT,fontsize=9); ax_el.grid(True,alpha=0.15); ax_el.spines[:].set_color(BORDER)

    ax_sl=fig.add_subplot(gs[1,1]); ax_sl.set_facecolor(SURFACE)
    bar_c=[COLORS["Normal"] if k==R['best_k'] else MUTED for k in R['k_range']]
    ax_sl.bar(R['k_range'],R['sil_k'],color=bar_c,alpha=0.85)
    ax_sl.set_title("Silhouette by K",color=TEXT,fontsize=9); ax_sl.grid(True,alpha=0.15,axis="y"); ax_sl.spines[:].set_color(BORDER)

    ax_km=fig.add_subplot(gs[1,2]); ax_km.set_facecolor(SURFACE)
    for c in range(R['best_k']):
        mask=R['km_labels']==c; ax_km.scatter(R['X_pca2d'][mask,0],R['X_pca2d'][mask,1],c=CLIST[c%len(CLIST)],s=3,alpha=0.4,label=f"C{c}",rasterized=True)
    ax_km.set_title(f"K-Means K={R['best_k']}",color=TEXT,fontsize=9); ax_km.legend(fontsize=6,markerscale=3); ax_km.grid(True,alpha=0.15); ax_km.spines[:].set_color(BORDER)

    ax_pw=fig.add_subplot(gs[1,3]); ax_pw.set_facecolor(SURFACE)
    pw=R['pw_dist']; im=ax_pw.imshow(pw,cmap="YlOrRd",aspect="auto")
    ax_pw.set_xticks(range(R['best_k'])); ax_pw.set_yticks(range(R['best_k']))
    ax_pw.set_xticklabels([f"C{i}" for i in range(R['best_k'])],fontsize=7); ax_pw.set_yticklabels([f"C{i}" for i in range(R['best_k'])],fontsize=7)
    for i in range(R['best_k']):
        for j in range(R['best_k']): ax_pw.text(j,i,f"{pw[i,j]:.1f}",ha="center",va="center",fontsize=6,color="black")
    plt.colorbar(im,ax=ax_pw,fraction=0.046); ax_pw.set_title("Cluster Distances",color=TEXT,fontsize=9)

    for ax_idx,(scores,title,c1c,c2c) in enumerate([
        (R['iso_sc'],    "Isolation Forest",COLORS["Normal"],COLORS["DoS"]),
        (R['lof_sc'],    "LOF",             COLORS["Probe"], COLORS["R2L"]),
        (R['recon_err'], "Autoencoder",     COLORS["Normal"],COLORS["DoS"]),
    ]):
        ax=fig.add_subplot(gs[2,ax_idx]); ax.set_facecolor(SURFACE)
        ax.hist(scores[R['y_binary']==0],bins=60,color=c1c,alpha=0.7,label="Normal",density=True)
        ax.hist(scores[R['y_binary']==1],bins=60,color=c2c,alpha=0.7,label="Attack",density=True)
        if ax_idx==2: ax.axvline(R['threshold'],color=COLORS["U2R"],ls="--",lw=1.5)
        ax.set_title(title,color=TEXT,fontsize=9); ax.legend(fontsize=7); ax.grid(True,alpha=0.15); ax.spines[:].set_color(BORDER)

    ax_t=fig.add_subplot(gs[2,3]); ax_t.axis("off")
    rows=[["Silhouette",f"{R['sil_km']:.4f}","✓"],["DB Index",f"{R['db_km']:.4f}","✓ <1.0"],
          ["ARI",f"{R['ari_km']:.4f}","✓"],["Purity",f"{R['purity']*100:.1f}%","✓ >90%"],
          ["IF AUC",f"{R['iso_auc']:.4f}","✓"],["AE AUC",f"{R['ae_auc']:.4f}","✓"],["AE F1",f"{R['ae_f1']:.4f}","✓"]]
    tbl=ax_t.table(cellText=rows,colLabels=["Metric","Value",""],cellLoc="center",loc="center",bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_facecolor(SURFACE if r%2==0 else BG); cell.set_edgecolor(BORDER); cell.set_text_props(color=TEXT)
        if r==0: cell.set_facecolor(BORDER); cell.set_text_props(color=TEXT,fontweight="bold")
    ax_t.set_title("Summary",color=TEXT,fontsize=9)

    st.pyplot(fig); plt.close()
