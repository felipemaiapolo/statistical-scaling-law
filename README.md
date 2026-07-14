# Statistical Scaling Law for LLM Skills

This project builds a latent-variable scaling model to study how large language model (LLM) capabilities grow with compute across diverse model families and benchmarks. It provides utilities to fit the model, sample family-level abilities, and reproduce the empirical studies reported in the paper.

## Repository structure

- `model.py` – PyTorch implementation of the latent-skill scaling model, fitting utilities, posterior sampling, and asymptotic variance estimation.
- `constants.py` – Guessing-rate lower bounds for each benchmark.
- `leaderboard_analysis.ipynb` – End-to-end notebook to load the Open LLM Leaderboard data, fit models with different latent dimensions, and generate the figures/tables in the paper.
- `data/` – Preprocessed leaderboard dumps (`df*.csv`) and intermediate arrays (`stes*.npy`, `models_0.npy`) used by the notebook.
- `models/` – Saved model outputs from previous runs.
- `plots/` – Figures produced by the analysis notebook.

## Getting started

1. **Install dependencies** (Python 3.13.12 recommended):
   ```bash
   pip install torch numpy pandas matplotlib seaborn tqdm joblib
   ```

2. **Inspect the data** (already included):
   - `data/df_full.csv`, `data/df_full_v1.csv`, `data/df_full_v2.csv` hold the combined benchmark scores and metadata.
   - Lower-bound constants for each benchmark are defined in `constants.py` and used when computing beta likelihoods.

3. **Run the analysis notebook**:
   - Open `leaderboard_analysis.ipynb` in Jupyter or VS Code.
   - Execute the cells to load the leaderboard data, choose the latent dimension (`Ks = [1,2,...,12]`), fit the scaling model via `ScalingLaw.fit`, and regenerate the plots.
   - The notebook demonstrates how to compare instruction-tuned and base models, build prediction intervals, and compute compute-optimal configurations.

## Using the model programmatically

```python
import numpy as np
import pandas as pd
from model import ScalingLaw

# X: design matrix with log-parameters/log-tokens features
# Y: benchmark scores in [0, 1]
# D: one-hot family indicators
# C: guessing-rate lower bounds for each benchmark
model = ScalingLaw(K=4)
model.fit(X, Y, D, C, B=5000, n_epochs=20000, device="cuda")

# Sample family-level abilities for posterior analyses
samples = model.sample_alpha(X, Y, D, C, fam_number=0, n_samples=2000)
```

The model supports missing benchmark entries (NaNs in `Y`) and uses Monte Carlo integration for the latent abilities during training.

## Data sources

The `data/` directory aggregates scores from both versions of the HuggingFace Open LLM Leaderboard. Fine-tuned instruction models are treated as distinct families from their base counterparts to capture family-specific scaling behavior.

## Reproducing paper figures

- Fit models for candidate latent dimensions using the notebook.
- Select the optimal dimension with AIC (Section 4 of the paper) and inspect `plots/` for the saved figures.
- Posterior sampling (`SampleAlpha`) and prediction intervals (`SampleParams`, `GetAsymptVar`) are demonstrated in the notebook to replicate the comparisons and uncertainty quantification shown in the paper.

## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.
