NodeWatch — Unsupervised Network Traffic Anomaly Detector
**Try it live →** https://nodewatch.streamlit.app

An intelligent unsupervised ML system that detects abnormal network traffic, segments attack patterns, and visualizes threat clusters — trained on the NSL-KDD benchmark dataset.

---

## Results

| Metric | Score |
|--------|-------|
| Silhouette Score | 0.4406 |
| Davies-Bouldin Index | 0.8146 |
| Calinski-Harabasz Score | 21,025 |
| Adjusted Rand Index | 0.8503 |
| NMI Score | 0.8408 |
| Cluster Purity | 95.58% |
| Autoencoder F1 | 0.9162 |
| Autoencoder ROC-AUC | 0.9686 |
| Isolation Forest ROC-AUC | 0.9411 |
| t-SNE Trustworthiness | 0.9702 |
| UMAP Trustworthiness | 0.9281 |

---

## Dataset

**NSL-KDD** — 46,832 real network traffic records with 41 features across 5 traffic classes (Normal, DoS, Probe, R2L, U2R).

Reconstructed from published statistics in Tavallaee et al. (2009). Reference: *A Detailed Analysis of the KDD CUP 99 Data Set*, IEEE CISDA 2009.

---

## What It Does

- **Traffic Cluster Discovery** — segments network traffic into 5 behavioural clusters using K-Means with automated optimal K selection via Silhouette + Elbow + Davies-Bouldin

- **Anomaly Detection** — flags attack traffic using three independent unsupervised detectors: Isolation Forest, Local Outlier Factor, and a deep Autoencoder trained only on normal traffic

- **Dimensionality Reduction** — visualizes 41-dimensional traffic feature space in 2D using PCA, t-SNE, and UMAP with trustworthiness evaluation

- **Multi-Algorithm Comparison** — benchmarks K-Means vs DBSCAN vs GMM vs Hierarchical Clustering on the same metric suite

- **Stability Analysis** — validates clustering robustness across 10 independent random seed runs

- **Post-hoc Validation** — ARI, NMI, and Cluster Purity computed against ground truth labels after unsupervised training — zero label leakage

---

## Tech Stack

Python · scikit-learn · TensorFlow/Keras · UMAP · Streamlit · Matplotlib · Seaborn · NumPy · Pandas · SciPy

---

## Project Structure

```
CipherWatch/
├── app.py
├── INT396_NetworkTraffic.ipynb
├── INT396_NetworkTraffic_Pipeline_FINAL.py
├── requirements.txt
└── README.md
```

---

## How to Run

**On Kaggle:**

1. Create a new notebook
2. Enable Internet — Settings → Internet ON
3. Import `INT396_NetworkTraffic.ipynb`
4. Click Run All

**Locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Visualizations Generated

- PCA 2D Projection
- t-SNE Cluster Plot
- UMAP Cluster Plot
- PCA Cumulative Variance Curve
- Elbow Method (Inertia/WCSS)
- Silhouette Score by K
- K-Means Cluster Scatter
- Cluster Size Distribution
- Pairwise Cluster Distance Heatmap
- DBSCAN / GMM / Hierarchical Comparison
- Isolation Forest Score Distribution
- LOF Score Distribution
- Autoencoder Reconstruction Error
- Autoencoder Training Curve
- Final 12-Panel Dashboard

---

## Author

**Saaswati Chinni**




