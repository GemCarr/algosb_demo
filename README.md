# Algosb demo
This repository contains two notebooks.

- notebooks/1_conformational_ensemble_analysis.ipynb

This notebook demonstrates seeding methods based on k-means++ seeding applied 
on pdb files, followed by landmark Multidimensional scaling using the selected
seeds as landmarks.

- notebooks/2_statistical_models.ipynb

This notebooks demonstrates basic distributions on the circle/torus, including
a d-Torus multivariate distribution based on circulass (circular copulas).
Then it loads mixture models of circula based distributions estimated on amino
acid torsion angles. Finally it shows how such mixture models can be 
conditioned on data subsets to obtain generative models for particular 
phenomenons.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
