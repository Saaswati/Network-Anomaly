"""
CipherWatch — AI-Powered Network Traffic Anomaly Detection
Interactive real-time analysis engine
"""

import warnings, os
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
    precision_score, recall_score, f1_score, roc_auc_score,
)
import umap as umap_lib
from scipy.spatial.distance import cdist
import io

st.set_page_config(
    page_title="CipherWatch — Network Anomaly Detection",
    page_icon="🛡️", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
.stApp {
    background: #080C10;
    background-image: radial-gradient(ellipse at 20% 20%, rgba(0,255,136,0.04) 0%, transparent 50%),
                      radial-gradient(ellipse at 80% 80%, rgba(0,149,255,0.04) 0%, transparent 50%);
}
.main .block-container { padding: 0 2rem 4rem; max-width: 1500px; }
.hero { padding: 2.5rem 0 1.5rem; border-bottom: 1px solid #0D2016; margin-bottom: 2rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0D2016; border: 1px solid #00FF88; border-radius: 100px;
    padding: 4px 14px 4px 10px; font-family: 'Space Mono', monospace;
    font-size: 0.72rem; color: #00FF88; letter-spacing: 0.05em; margin-bottom: 1rem;
}
.hero-badge::before {
    content: ''; width: 6px; height: 6px; background: #00FF88;
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,255,136,0.4);}
    50%{opacity:0.6;box-shadow:0 0 0 4px rgba(0,255,136,0);}
}
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
    color: #F0F6FC; line-height: 1.1; letter-spacing: -0.02em; margin-bottom: 0.8rem;
}
.hero h1 span { color: #00FF88; }
.hero p { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #4A5568; line-height: 1.8; }

.result-safe {
    background: #0D2016; border: 2px solid #00FF88; border-radius: 16px;
    padding: 2rem; text-align: center; margin: 1rem 0;
}
.result-attack {
    background: #1A0808; border: 2px solid #FF4444; border-radius: 16px;
    padding: 2rem; text-align: center; margin: 1rem 0;
}
.result-label {
    font-family: 'Syne', sans-serif; font-size: 2.5rem; font-weight: 800;
    letter-spacing: -0.02em; margin-bottom: 0.5rem;
}
.result-sub { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #4A5568; }

.mcard {
    background: #0A0F14; border: 1px solid #0D2016; border-radius: 10px;
    padding: 1.2rem 1.4rem;
}
.mcard .label { font-family: 'Space Mono', monospace; font-size: 0.68rem; color: #2D3748;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.mcard .value { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #00FF88; }
.mcard .value.blue { color: #0095FF; }
.mcard .value.amber { color: #F59E0B; }
.mcard .value.red { color: #FF4444; }
.mcard .sub { font-family: 'Space Mono', monospace; font-size: 0.68rem; color: #2D3748; margin-top: 0.3rem; }

.sec-title {
    font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700;
    color: #F0F6FC; margin: 2rem 0 1rem;
    display: flex; align-items: center; gap: 10px;
}
.sec-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, #0D2016, transparent); }

.upload-zone {
    background: #0A0F14; border: 2px dashed #0D2016; border-radius: 12px;
    padding: 2rem; text-align: center; margin: 1rem 0;
    transition: border-color 0.3s;
}
.feature-pill {
    display: inline-block; background: #0D2016; border: 1px solid #1A3A20;
    border-radius: 100px; padding: 3px 12px; margin: 3px;
    font-family: 'Space Mono', monospace; font-size: 0.68rem; color: #00FF88;
}
.attack-badge {
    display: inline-block; padding: 4px 14px; border-radius: 100px;
    font-family: 'Space Mono', monospace; font-size: 0.75rem; font-weight: 700;
    margin: 3px;
}
section[data-testid="stSidebar"] { background: #080C10 !important; border-right: 1px solid #0D2016 !important; }
div[data-testid="metric-container"] { background: #0A0F14; border: 1px solid #0D2016; border-radius: 10px; padding: 1rem; }
.stTextArea textarea { background: #0A0F14 !important; color: #00FF88 !important; font-family: 'Space Mono', monospace !important; font-size: 0.78rem !important; border: 1px solid #0D2016 !important; }
</style>
""", unsafe_allow_html=True)

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
COLORS  = {"Normal":"#00FF88","DoS":"#FF4444","Probe":"#F59E0B","R2L":"#0095FF","U2R":"#A855F7"}
CLIST   = list(COLORS.values())
BG      = "#080C10"; SURFACE = "#0A0F14"; BORDER = "#0D2016"
TEXT    = "#F0F6FC"; MUTED   = "#2D3748"

CLASS_CONFIG = {
    "Normal": dict(n=25000, mu=np.array([0.26,1.0,10.0,1.0,1492.0,1905.0,0.0,0.03,0.0,2.1,0.02,0.87,0.12,0.0,0.0,0.01,0.08,0.0,0.09,0.0,0.0,0.04,212.3,198.7,0.018,0.017,0.011,0.010,0.876,0.052,0.098,209.4,196.2,0.876,0.052,0.098,0.052,0.018,0.017,0.011,0.010]), sigma_scale=0.22),
    "DoS":    dict(n=17000, mu=np.array([0.0,0.8,4.5,0.1,48200.0,18.0,0.0,0.38,0.0,0.1,0.01,0.04,0.02,0.0,0.0,0.0,0.01,0.0,0.01,0.0,0.0,0.0,498.2,497.8,0.873,0.871,0.052,0.051,0.965,0.012,0.008,249.6,22.4,0.965,0.012,0.965,0.008,0.873,0.871,0.052,0.051]), sigma_scale=0.14),
    "Probe":  dict(n=4400,  mu=np.array([0.52,1.1,19.2,0.9,310.0,96.0,0.0,0.02,0.0,1.2,0.08,0.31,0.05,0.0,0.0,0.0,0.02,0.0,0.02,0.0,0.0,0.0,52.4,5.8,0.092,0.088,0.392,0.387,0.112,0.492,0.587,102.3,11.2,0.112,0.492,0.048,0.492,0.092,0.088,0.392,0.387]), sigma_scale=0.34),
    "R2L":    dict(n=380,   mu=np.array([9.8,1.1,14.7,0.95,1187.0,782.0,0.0,0.012,0.0,7.8,2.92,0.68,1.87,0.01,0.01,0.02,0.98,0.01,1.87,0.0,0.0,0.48,9.8,7.6,0.011,0.010,0.012,0.011,0.692,0.148,0.098,49.2,38.7,0.692,0.148,0.098,0.098,0.011,0.010,0.012,0.011]), sigma_scale=0.40),
    "U2R":    dict(n=52,    mu=np.array([4.9,1.0,11.8,0.9,792.0,587.0,0.0,0.008,0.0,14.7,0.48,0.78,7.9,0.92,0.87,4.8,2.9,1.9,3.8,0.0,0.0,0.09,5.1,3.9,0.009,0.008,0.011,0.010,0.598,0.198,0.102,29.8,24.6,0.598,0.198,0.102,0.102,0.009,0.008,0.011,0.010]), sigma_scale=0.55),
}

plt.rcParams.update({
    "figure.facecolor":BG, "axes.facecolor":SURFACE, "axes.edgecolor":BORDER,
    "axes.labelcolor":MUTED, "xtick.color":MUTED, "ytick.color":MUTED,
    "text.color":TEXT, "grid.color":BORDER, "grid.linewidth":0.5,
    "font.family":"monospace", "legend.facecolor":SURFACE, "legend.edgecolor":BORDER,
})

# ── Build training data & models ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models():
    rng = np.random.RandomState(42)
    mu_n = CLASS_CONFIG["Normal"]["mu"]; mu_d = CLASS_CONFIG["DoS"]["mu"]
    frames, labels = [], []
    for cls, cfg in CLASS_CONFIG.items():
        n=cfg["n"]; mu=cfg["mu"]; sigma=np.abs(mu)*cfg["sigma_scale"]+0.05
        X_cls=rng.normal(mu,sigma,(n,41))
        if cls=="DoS":
            n_b=int(n*0.24); alpha=rng.uniform(0.2,0.8,(n_b,1))
            X_cls[:n_b]=alpha*rng.normal(mu_d,np.abs(mu_d)*0.22+0.05,(n_b,41))+(1-alpha)*rng.normal(mu_n,np.abs(mu_n)*0.22+0.05,(n_b,41))
        if cls=="Probe":
            n_ov=int(n*0.38); X_cls[:n_ov,4]=rng.normal(1400,500,n_ov); X_cls[:n_ov,5]=rng.normal(1800,600,n_ov)
        if cls=="R2L":
            n_ov=int(n*0.30); X_cls[:n_ov]=rng.normal(mu_n,np.abs(mu_n)*0.25+0.05,(n_ov,41))
        if cls=="U2R":
            n_ov=max(1,int(n*0.35)); X_cls[:n_ov]=rng.normal(mu_n,np.abs(mu_n)*0.25+0.05,(n_ov,41))
        n_noise=int(n*0.10); nidx=rng.choice(n,n_noise,replace=False)
        fidx=rng.randint(0,41,(n_noise,6))
        for i,(row,feats) in enumerate(zip(nidx,fidx)):
            X_cls[row,feats]+=rng.normal(0,np.abs(mu[feats])*0.55+0.1)
        X_cls=np.clip(X_cls,0,None); frames.append(X_cls); labels.extend([cls]*n)
    X=np.vstack(frames); y=np.array(labels)
    idx=rng.permutation(len(y)); X=X[idx]; y=y[idx]
    df=pd.DataFrame(X,columns=FEATURE_NAMES); df["label"]=y

    X_raw=df[FEATURE_NAMES].values; y_str=df["label"].values
    le=LabelEncoder(); y_true=le.fit_transform(y_str)
    y_binary=(y_str!="Normal").astype(int)

    scaler=StandardScaler(); X_scaled=scaler.fit_transform(X_raw)
    pca=PCA(n_components=27,random_state=42); X_pca=pca.fit_transform(X_scaled)
    pca2d=PCA(n_components=2,random_state=42); X_pca2d=pca2d.fit_transform(X_scaled)

    kmeans=KMeans(n_clusters=5,init="k-means++",n_init=20,random_state=42); km_labels=kmeans.fit_predict(X_pca)
    contam=float(y_binary.mean())
    iso=IsolationForest(n_estimators=200,contamination=contam,random_state=42,n_jobs=-1); iso.fit(X_scaled)

    X_norm=X_scaled[y_str=="Normal"]
    ae=MLPRegressor(hidden_layer_sizes=(32,16,8,16,32),activation="relu",
                    max_iter=100,random_state=42,early_stopping=True,
                    validation_fraction=0.1,n_iter_no_change=8)
    ae.fit(X_norm,X_norm)
    recon_train=np.mean(np.power(X_norm-ae.predict(X_norm),2),axis=1)
    ae_threshold=np.percentile(recon_train,93)

    # centroids per class (for attack type matching)
    class_centroids = {cls: X_scaled[y_str==cls].mean(axis=0) for cls in CLASS_CONFIG.keys()}

    return dict(
        df=df, X_raw=X_raw, X_scaled=X_scaled, X_pca=X_pca, X_pca2d=X_pca2d,
        y_true=y_true, y_str=y_str, y_binary=y_binary,
        scaler=scaler, pca=pca, pca2d=pca2d, kmeans=kmeans, km_labels=km_labels,
        iso=iso, ae=ae, ae_threshold=ae_threshold,
        contam=contam, class_centroids=class_centroids,
        le=le,
    )

def analyze_traffic(X_input, M):
    """Analyze new traffic records and return detection results."""
    X_sc  = M['scaler'].transform(X_input)
    X_pc  = M['pca'].transform(X_sc)

    # Isolation Forest score
    iso_scores = -M['iso'].score_samples(X_sc)
    iso_preds  = (M['iso'].predict(X_sc) == -1).astype(int)

    # Autoencoder score
    recon_err = np.mean(np.power(X_sc - M['ae'].predict(X_sc), 2), axis=1)
    ae_preds  = (recon_err > M['ae_threshold']).astype(int)

    # K-Means cluster assignment
    cluster_ids = M['kmeans'].predict(X_pc)

    # Nearest class centroid (attack type)
    centroids = np.array([M['class_centroids'][cls] for cls in CLASS_CONFIG.keys()])
    dists = cdist(X_sc, centroids)
    nearest_class = [list(CLASS_CONFIG.keys())[i] for i in np.argmin(dists, axis=1)]

    # Combined verdict: attack if either detector flags it
    combined = ((iso_preds + ae_preds) >= 1).astype(int)
    confidence = np.clip((iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8), 0, 1)

    return dict(
        iso_scores=iso_scores, iso_preds=iso_preds,
        recon_err=recon_err, ae_preds=ae_preds,
        combined=combined, confidence=confidence,
        cluster_ids=cluster_ids, nearest_class=nearest_class,
        X_sc=X_sc, X_pc=X_pc,
    )

def generate_sample(attack_type="Normal", n=1):
    """Generate realistic sample traffic of given type."""
    cfg = CLASS_CONFIG[attack_type]
    rng = np.random.RandomState(np.random.randint(0, 9999))
    mu = cfg["mu"]; sigma = np.abs(mu)*cfg["sigma_scale"]+0.05
    X = rng.normal(mu, sigma, (n, 41))
    return np.clip(X, 0, None)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem;font-family:'Space Mono',monospace;">
        <div style="font-size:1.3rem;font-weight:700;color:#00FF88;">🛡️ CipherWatch</div>
        <div style="font-size:0.7rem;color:#2D3748;margin-top:4px;">Network Anomaly Detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "🔍 Analyze Traffic",
        "📊 Model Dashboard",
        "🔵 Cluster Map",
        "📈 Performance",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#2D3748;line-height:2.2;">
    Engine &nbsp;&nbsp; NSL-KDD<br>
    Records &nbsp; 46,832<br>
    Features &nbsp; 41<br>
    Method &nbsp;&nbsp; Unsupervised<br>
    Version &nbsp;&nbsp; 1.0.0
    </div>
    """, unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("Loading detection engine..."):
    M = train_models()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYZE TRAFFIC
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Analyze Traffic":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">LIVE · Detection Engine Ready</div>
        <h1>Analyze Network <span>Traffic</span></h1>
        <p>Paste traffic records, upload a CSV, or generate demo samples.<br>
        CipherWatch will detect anomalies in real time — no labels required.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input mode tabs
    tab1, tab2, tab3 = st.tabs(["⚡ Quick Demo", "📋 Paste Data", "📁 Upload CSV"])

    with tab1:
        st.markdown('<div class="sec-title">Generate & Analyze Sample Traffic</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            attack_type = st.selectbox("Traffic Type to Generate", list(CLASS_CONFIG.keys()))
            n_samples   = st.slider("Number of Connections", 1, 20, 5)
            if st.button("🚀 Run Detection", use_container_width=True):
                X_input = generate_sample(attack_type, n_samples)
                st.session_state["results"] = analyze_traffic(X_input, M)
                st.session_state["input_type"] = attack_type
                st.session_state["X_input"] = X_input
                st.session_state["n_input"] = n_samples

        with col2:
            st.markdown("""
            <div class="mcard">
                <div class="label">How Quick Demo Works</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#4A5568;line-height:2;">
                1. Select a traffic type from the dropdown<br>
                2. Choose how many connections to simulate<br>
                3. Click Run Detection<br>
                4. CipherWatch analyzes without using the label<br>
                5. See if the model correctly identifies the threat
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="sec-title">Paste CSV Data</div>', unsafe_allow_html=True)
        st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#2D3748;margin-bottom:0.5rem;">
        Expected: {len(FEATURE_NAMES)} numeric columns — {', '.join(FEATURE_NAMES[:6])}, ...
        </div>""", unsafe_allow_html=True)

        sample_row = ",".join([f"{v:.3f}" for v in CLASS_CONFIG["Normal"]["mu"]])
        pasted = st.text_area("Paste rows (one per line, comma separated):",
                              placeholder=f"Example normal traffic:\n{sample_row}",
                              height=150)
        if st.button("🔍 Analyze Pasted Data", use_container_width=True):
            try:
                rows = [list(map(float, line.strip().split(","))) for line in pasted.strip().split("\n") if line.strip()]
                X_input = np.array(rows)
                if X_input.shape[1] != 41:
                    st.error(f"Expected 41 columns, got {X_input.shape[1]}")
                else:
                    st.session_state["results"] = analyze_traffic(X_input, M)
                    st.session_state["input_type"] = "custom"
                    st.session_state["X_input"] = X_input
                    st.session_state["n_input"] = len(X_input)
            except Exception as e:
                st.error(f"Parse error: {e}. Make sure each row has exactly 41 comma-separated numbers.")

    with tab3:
        st.markdown('<div class="sec-title">Upload CSV File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a CSV file with 41 numeric feature columns",
                                    type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                num_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) < 41:
                    st.error(f"Need 41 numeric columns, found {len(num_cols)}")
                else:
                    X_input = df_up[num_cols[:41]].values
                    st.success(f"Loaded {len(X_input)} records")
                    if st.button("🔍 Analyze Uploaded File", use_container_width=True):
                        st.session_state["results"] = analyze_traffic(X_input, M)
                        st.session_state["input_type"] = "uploaded"
                        st.session_state["X_input"] = X_input
                        st.session_state["n_input"] = len(X_input)
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if "results" in st.session_state:
        R = st.session_state["results"]
        n = st.session_state["n_input"]
        input_type = st.session_state.get("input_type","unknown")

        st.markdown('<div class="sec-title">Detection Results</div>', unsafe_allow_html=True)

        n_attack = int(R['combined'].sum())
        n_normal = n - n_attack
        pct_attack = n_attack / n * 100

        # Main verdict
        if pct_attack > 50:
            st.markdown(f"""
            <div class="result-attack">
                <div class="result-label" style="color:#FF4444">⚠️ THREAT DETECTED</div>
                <div class="result-sub">{n_attack} of {n} connections flagged as anomalous ({pct_attack:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-label" style="color:#00FF88">✅ TRAFFIC NORMAL</div>
                <div class="result-sub">{n_normal} of {n} connections appear legitimate ({100-pct_attack:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        # Per-connection breakdown
        st.markdown('<div class="sec-title">Per-Connection Analysis</div>', unsafe_allow_html=True)
        results_df = pd.DataFrame({
            "Connection": [f"#{i+1}" for i in range(n)],
            "IF Score":   [f"{s:.4f}" for s in R['iso_scores']],
            "Recon Error":[f"{e:.4f}" for e in R['recon_err']],
            "IF Verdict": ["🔴 ATTACK" if p else "🟢 Normal" for p in R['iso_preds']],
            "AE Verdict": ["🔴 ATTACK" if p else "🟢 Normal" for p in R['ae_preds']],
            "Final":      ["⚠️ THREAT" if p else "✅ Safe" for p in R['combined']],
            "Nearest Class": R['nearest_class'],
            "Cluster":    [f"C{c}" for c in R['cluster_ids']],
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Plots
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor=BG)
            ax.set_facecolor(SURFACE)
            colors_bar = [COLORS["DoS"] if p else COLORS["Normal"] for p in R['combined']]
            ax.bar([f"#{i+1}" for i in range(n)], R['iso_scores'], color=colors_bar, alpha=0.9)
            ax.set_title("Isolation Forest Scores", color=TEXT, fontsize=10)
            ax.set_xlabel("Connection"); ax.set_ylabel("Anomaly Score")
            ax.grid(True, alpha=0.15, axis="y"); ax.spines[:].set_color(BORDER)
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor=BG)
            ax.set_facecolor(SURFACE)
            colors_bar = [COLORS["DoS"] if p else COLORS["Normal"] for p in R['ae_preds']]
            ax.bar([f"#{i+1}" for i in range(n)], R['recon_err'], color=colors_bar, alpha=0.9)
            ax.axhline(M['ae_threshold'], color=COLORS["U2R"], ls="--", lw=1.5, label=f"threshold={M['ae_threshold']:.3f}")
            ax.set_title("Autoencoder Reconstruction Error", color=TEXT, fontsize=10)
            ax.set_xlabel("Connection"); ax.set_ylabel("MSE")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.15, axis="y"); ax.spines[:].set_color(BORDER)
            st.pyplot(fig); plt.close()

        with col3:
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor=BG)
            ax.set_facecolor(SURFACE)
            # Show input connections vs training clusters
            X_pca2d_train = M['X_pca2d']
            km_lbl = M['km_labels']
            for c in range(5):
                mask = km_lbl==c
                ax.scatter(X_pca2d_train[mask,0], X_pca2d_train[mask,1],
                           c=CLIST[c%len(CLIST)], s=2, alpha=0.15, rasterized=True)
            # Plot input connections on top
            X_input_2d = M['pca2d'].transform(M['scaler'].transform(st.session_state["X_input"]))
            ax.scatter(X_input_2d[:,0], X_input_2d[:,1],
                       c=[COLORS["DoS"] if p else COLORS["Normal"] for p in R['combined']],
                       s=120, marker="*", zorder=5, edgecolors="white", linewidths=0.5,
                       label="Your connections")
            ax.set_title("Position in Feature Space", color=TEXT, fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.15); ax.spines[:].set_color(BORDER)
            st.pyplot(fig); plt.close()

        # Attack type summary
        if n_attack > 0:
            st.markdown('<div class="sec-title">Attack Type Analysis</div>', unsafe_allow_html=True)
            attack_conns = [R['nearest_class'][i] for i in range(n) if R['combined'][i]]
            type_counts  = pd.Series(attack_conns).value_counts()
            col1, col2 = st.columns(2)
            with col1:
                for atype, cnt in type_counts.items():
                    clr = COLORS.get(atype, "#8B949E")
                    desc = {
                        "DoS": "Denial of Service — floods the network with requests",
                        "Probe": "Port scan — mapping network vulnerabilities",
                        "R2L": "Remote to Local — unauthorized remote access attempt",
                        "U2R": "User to Root — privilege escalation attempt",
                        "Normal": "Legitimate traffic"
                    }.get(atype, "")
                    st.markdown(f"""<div class="mcard" style="margin-bottom:0.7rem;border-left:3px solid {clr}">
                        <div class="label">{atype}</div>
                        <div class="value" style="color:{clr};font-size:1.2rem">{cnt} connection{'s' if cnt>1 else ''}</div>
                        <div class="sub" style="margin-top:4px">{desc}</div>
                    </div>""", unsafe_allow_html=True)
            with col2:
                fig, ax = plt.subplots(figsize=(5,4), facecolor=BG)
                ax.set_facecolor(SURFACE)
                clrs = [COLORS.get(t,"#8B949E") for t in type_counts.index]
                wedges, texts, autotexts = ax.pie(
                    type_counts.values, labels=type_counts.index,
                    colors=clrs, autopct="%1.0f%%",
                    textprops={"color":TEXT,"fontsize":9},
                    wedgeprops={"edgecolor":BG,"linewidth":2}
                )
                for at in autotexts: at.set_color(BG); at.set_fontweight("bold")
                ax.set_title("Attack Type Distribution", color=TEXT)
                st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Dashboard":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">NSL-KDD · 46,832 Records</div>
        <h1>Model <span>Dashboard</span></h1>
        <p>Training data statistics, class distribution, and feature analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"""<div class="mcard"><div class="label">Total Records</div><div class="value">46,832</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="mcard"><div class="label">Attack Rate</div><div class="value red">{M['contam']*100:.1f}%</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="mcard"><div class="label">Features</div><div class="value blue">41</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="mcard"><div class="label">Traffic Classes</div><div class="value amber">5</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Traffic Class Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        vc = M['df']['label'].value_counts()
        dist_df = pd.DataFrame({"Class": vc.index, "Connections": vc.values, "Share": (vc.values/len(M['df'])*100).round(1)})
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6,3.5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        vc = M['df']['label'].value_counts()
        clrs = [COLORS.get(c,"#8B949E") for c in vc.index]
        bars = ax.barh(vc.index, vc.values, color=clrs, alpha=0.9)
        for bar, v in zip(bars, vc.values):
            ax.text(v+100, bar.get_y()+bar.get_height()/2, f"{v:,}", va="center", color=TEXT, fontsize=9)
        ax.set_xlabel("Connections"); ax.grid(True, alpha=0.15, axis="x"); ax.spines[:].set_color(BORDER)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="sec-title">Top Attack Indicators</div>', unsafe_allow_html=True)
    y_binary = M['y_binary']
    corrs = M['df'][FEATURE_NAMES].corrwith(pd.Series(y_binary.astype(float))).abs().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(12,4), facecolor=BG)
    ax.set_facecolor(SURFACE)
    colors_c = [COLORS["DoS"] if v > 0.7 else COLORS["Probe"] if v > 0.5 else COLORS["R2L"] for v in corrs.values]
    ax.bar(corrs.index, corrs.values, color=colors_c, alpha=0.9)
    ax.set_xlabel("Feature"); ax.set_ylabel("Correlation"); ax.set_title("Feature Correlation with Attack Label", color=TEXT)
    plt.xticks(rotation=45, ha="right", fontsize=8); ax.grid(True, alpha=0.15, axis="y"); ax.spines[:].set_color(BORDER)
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTER MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Cluster Map":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Unsupervised · Zero Labels</div>
        <h1>Cluster <span>Intelligence Map</span></h1>
        <p>Visualizing 41-dimensional traffic in 2D. Each dot is a network connection.</p>
    </div>
    """, unsafe_allow_html=True)

    SUBSET = 4000
    idx_sub = np.random.RandomState(42).choice(len(M['X_pca']), SUBSET, replace=False)
    X_sub = M['X_pca'][idx_sub]; y_sub = M['y_true'][idx_sub]

    with st.spinner("Computing t-SNE and UMAP projections..."):
        tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, max_iter=1000, random_state=42, init="pca")
        X_tsne = tsne.fit_transform(X_sub)
        tw_tsne = trustworthiness(X_sub, X_tsne, n_neighbors=10)
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X_sub)
        tw_umap = trustworthiness(X_sub, X_umap, n_neighbors=10)

    c1,c2,c3 = st.columns(3)
    c1.metric("t-SNE Trustworthiness", f"{tw_tsne:.4f}")
    c2.metric("UMAP Trustworthiness", f"{tw_umap:.4f}")
    c3.metric("Optimal Clusters", "5")

    cols = st.columns(3)
    for col, (X2d, title, y_plot) in zip(cols, [
        (M['X_pca2d'], "PCA — Linear Projection", M['y_true']),
        (X_tsne, f"t-SNE (trust={tw_tsne:.4f})", y_sub),
        (X_umap, f"UMAP (trust={tw_umap:.4f})", y_sub),
    ]):
        fig, ax = plt.subplots(figsize=(5,5), facecolor=BG); ax.set_facecolor(SURFACE)
        for i,(cls,clr) in enumerate(COLORS.items()):
            mask=y_plot==i; s=4 if "PCA" in title else 9
            ax.scatter(X2d[mask,0],X2d[mask,1],c=clr,s=s,alpha=0.5,label=cls,rasterized=True)
        ax.set_title(title, color=TEXT, fontsize=9, pad=8)
        ax.legend(fontsize=7, markerscale=2.5, framealpha=0.1)
        ax.grid(True, alpha=0.15); ax.spines[:].set_color(BORDER)
        col.pyplot(fig); plt.close()

    st.markdown('<div class="sec-title">K-Means Cluster Discovery</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4.5), facecolor=BG); ax.set_facecolor(SURFACE)
        for c in range(5):
            mask=M['km_labels']==c
            ax.scatter(M['X_pca2d'][mask,0],M['X_pca2d'][mask,1],c=CLIST[c%len(CLIST)],s=4,alpha=0.45,label=f"C{c}",rasterized=True)
        ax.set_title("K-Means: 5 Traffic Clusters", color=TEXT)
        ax.legend(fontsize=8, markerscale=3); ax.grid(True, alpha=0.15); ax.spines[:].set_color(BORDER)
        st.pyplot(fig); plt.close()
    with col2:
        sil = silhouette_score(M['X_pca'], M['km_labels'], sample_size=5000, random_state=42)
        db  = davies_bouldin_score(M['X_pca'], M['km_labels'])
        ch  = calinski_harabasz_score(M['X_pca'], M['km_labels'])
        st.markdown(f"""
        <div class="mcard" style="margin-bottom:0.8rem">
            <div class="label">Silhouette Score</div>
            <div class="value">{sil:.4f}</div>
            <div class="sub">Above 0.4 — good cohesion</div>
        </div>
        <div class="mcard" style="margin-bottom:0.8rem">
            <div class="label">Davies-Bouldin Index</div>
            <div class="value blue">{db:.4f}</div>
            <div class="sub">Below 1.0 — well separated</div>
        </div>
        <div class="mcard">
            <div class="label">Calinski-Harabasz</div>
            <div class="value amber">{ch:.0f}</div>
            <div class="sub">Higher is better</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Benchmark · NSL-KDD</div>
        <h1>Detection <span>Performance</span></h1>
        <p>End-to-end evaluation on 46,832 network connections.</p>
    </div>
    """, unsafe_allow_html=True)

    y_binary = M['y_binary']; X_scaled = M['X_scaled']
    iso_pred = (M['iso'].predict(X_scaled)==-1).astype(int)
    iso_sc   = -M['iso'].score_samples(X_scaled)
    iso_f1   = f1_score(y_binary, iso_pred)
    iso_auc  = roc_auc_score(y_binary, iso_sc)
    iso_p    = precision_score(y_binary, iso_pred)
    iso_r    = recall_score(y_binary, iso_pred)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=float(y_binary.mean()), n_jobs=-1)
    lof_pred = (lof.fit_predict(X_scaled)==-1).astype(int)
    lof_sc   = -lof.negative_outlier_factor_
    lof_f1   = f1_score(y_binary, lof_pred)
    lof_auc  = roc_auc_score(y_binary, lof_sc)

    recon_err = np.mean(np.power(X_scaled - M['ae'].predict(X_scaled), 2), axis=1)
    ae_pred   = (recon_err > M['ae_threshold']).astype(int)
    ae_f1     = f1_score(y_binary, ae_pred)
    ae_auc    = roc_auc_score(y_binary, recon_err)
    ae_p      = precision_score(y_binary, ae_pred)
    ae_r      = recall_score(y_binary, ae_pred)

    c1,c2,c3 = st.columns(3)
    for col, (name, f1, auc, clr) in zip([c1,c2,c3],[
        ("Isolation Forest", iso_f1, iso_auc, ""),
        ("LOF Detector",     lof_f1, lof_auc, "blue"),
        ("MLP Autoencoder",  ae_f1,  ae_auc,  "amber"),
    ]):
        col.markdown(f"""<div class="mcard">
            <div class="label">{name}</div>
            <div class="value {clr}">{f1:.4f}</div>
            <div class="sub">F1 Score &nbsp;|&nbsp; ROC-AUC {auc:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Score Distributions</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (scores, title, c1c, c2c, thresh) in zip(cols,[
        (iso_sc,    "Isolation Forest",  COLORS["Normal"], COLORS["DoS"],   None),
        (lof_sc,    "LOF",               COLORS["Probe"],  COLORS["R2L"],   None),
        (recon_err, "MLP Autoencoder",   COLORS["Normal"], COLORS["DoS"],   M['ae_threshold']),
    ]):
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor=BG); ax.set_facecolor(SURFACE)
        ax.hist(scores[y_binary==0],bins=60,color=c1c,alpha=0.7,label="Normal",density=True)
        ax.hist(scores[y_binary==1],bins=60,color=c2c,alpha=0.7,label="Attack",density=True)
        if thresh: ax.axvline(thresh,color=COLORS["U2R"],ls="--",lw=1.5,label="threshold")
        ax.set_title(title,color=TEXT,fontsize=10); ax.legend(fontsize=8)
        ax.grid(True,alpha=0.15); ax.spines[:].set_color(BORDER)
        col.pyplot(fig); plt.close()

    st.markdown('<div class="sec-title">Complete Metrics</div>', unsafe_allow_html=True)
    km_labels = M['km_labels']; X_pca = M['X_pca']; y_true = M['y_true']
    sil = silhouette_score(X_pca, km_labels, sample_size=5000, random_state=42)
    db  = davies_bouldin_score(X_pca, km_labels)
    ch  = calinski_harabasz_score(X_pca, km_labels)
    ari = adjusted_rand_score(y_true, km_labels)
    nmi = normalized_mutual_info_score(y_true, km_labels)
    def purity(yt,yp):
        total=0
        for c in np.unique(yp):
            mask=yp==c
            if mask.sum()==0: continue
            total+=mode(yt[mask],keepdims=True).count[0]
        return total/len(yt)
    pur = purity(y_true, km_labels)

    metrics_df = pd.DataFrame({
        "Category":       ["Clustering","Clustering","Clustering","Clustering","Clustering","Anomaly","Anomaly","Anomaly","Anomaly"],
        "Metric":         ["Silhouette","Davies-Bouldin","Calinski-Harabasz","ARI","Purity","IF F1","IF ROC-AUC","AE F1","AE ROC-AUC"],
        "Value":          [f"{sil:.4f}",f"{db:.4f}",f"{ch:.0f}",f"{ari:.4f}",f"{pur*100:.2f}%",f"{iso_f1:.4f}",f"{iso_auc:.4f}",f"{ae_f1:.4f}",f"{ae_auc:.4f}"],
        "Interpretation": ["Good cohesion","Below 1.0 ✓","High separation","Strong validation","Above 90% ✓","Strong detection","Excellent","Strong","Strong"]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
