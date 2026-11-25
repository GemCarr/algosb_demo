# Algosb demo
This repository contains two notebooks.

- notebooks/1_statistical_models.ipynb

This notebook illustrates statistical models on amino acid torsion angles.

1. **Distributions on the torus**: classical distributions on the circle and torus are shown. Mixture models estimated on Alanine and Proline torsion angles are analyzed. These mixture models use circula based multivariate distributions as components.
2. **Conditioning on capping mechanism**: a mixture model estimated on Serine-Alanine amino-acid pairs is conditioned on a specific data subset. We select data representing non-helix-to-helix transition, as these display a phenomenon known as capping which has a noticeable influence on the side chain of Serine. Through a reweighting of the SERALA mixture model, we can obtain a conditioned density representative of the capping phenomenon.

- notebooks/2_conformational_ensemble_analysis.ipynb

This notebook illustrates how can proteins conformations can be clustered with k-means++ and visualized through multidimensional scaling.

1. **Kmeans++ seeding based clustering**: $k$ representative conformations (*seeds*) are selected with the Kmeans++ seeding (EGD), refined by EGD reselection (EGD-EGD); each conformation is assigned to its nearest seed, which defines $k$ clusters. Scanning $k=1,\dots,k_{\max}$ and locating the elbow of the curve $k \mapsto \sum_i \min_j d(c_i, s_j)$ suggests a number of clusters.
2. **Embedding via MDS**: the ensemble is embedded in 2D/3D by landmark multi-dimensional scaling, the seeds being the landmarks.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
