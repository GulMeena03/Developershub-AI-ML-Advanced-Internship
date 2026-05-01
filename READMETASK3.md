# Multimodal Housing Price Prediction – Image + Tabular Data

## Objective

The primary objective of this project is to develop and evaluate a multimodal machine learning model that predicts house prices by simultaneously processing **tabular features** (e.g., square footage, number of bedrooms, age, neighborhood) and **visual information** from synthetic house images. The goal is to demonstrate the effectiveness of combining structured data with deep visual features to improve regression accuracy compared to using either modality alone.

## Methodology / Approach

### 1. Dataset Generation
- A synthetic dataset of **800 houses** was created.
- Tabular features: 10 attributes including `sqft`, `bedrooms`, `bathrooms`, `age_years`, `garage_spots`, `lot_size`, `floors`, `pool`, `neighborhood`, `house_type`.
- Target: `price` (in USD), computed with a non‑linear rule plus noise.
- For each house, a synthetic **RGB image** (128×128) was generated, where visual properties (e.g., roof color, presence of a pool) correlate with the price.

### 2. Preprocessing
- Tabular features were standardized (StandardScaler).
- Categorical variables (`neighborhood`, `house_type`) were label‑encoded.
- Images were resized, normalized using ImageNet statistics, and augmented (random crop, horizontal flip, color jitter) for training.
- Data split: 560 train / 120 validation / 120 test.

### 3. Model Architecture
- **Image branch**: Pretrained ResNet‑18 (early layers frozen) with a projection head (256‑d output).
- **Tabular branch**: Small MLP with batch normalization and dropout (128‑d output).
- **Fusion**: Concatenation of image and tabular embeddings followed by a 3‑layer MLP (256 → 64 → 1) to predict log‑transformed price.
- Loss function: Huber loss (delta=0.5) for robustness.

### 4. Training
- Optimizer: AdamW (lr=3e-4, weight decay=1e-4).
- Cosine annealing learning rate schedule.
- Early stopping (patience=6) based on validation loss.
- Model selection: best checkpoint according to lowest validation loss.

### 5. Evaluation
- Metrics: MAE, RMSE, MAPE, R².
- Ablation study: compared multimodal model against image‑only and tabular‑only variants.
- Permutation feature importance for tabular features.
- Visual analysis: residual plots, prediction vs. actual, sample image predictions.

## Key Results & Observations

**Test Set Performance (Multimodal Model)**
| Metric       | Value      |
|--------------|------------|
| MAE          | $77,003    |
| RMSE         | $98,465    |
| MAPE         | 16.55%     |
| R²           | 0.8871     |

**Ablation Study (15‑epoch quick training)**
| Model             | MAE (K$) | RMSE (K$) | R²     |
|-------------------|----------|-----------|--------|
| Image Only        | 277.46   | 319.42    | -0.188 |
| Tabular Only      | 476.01   | 854.73    | -7.507 |
| Multimodal (full) | 77.00    | 98.46     | 0.887  |

**Observations**
- The multimodal model substantially outperforms both single‑modality baselines, showing that combining visual and tabular information yields more accurate price predictions.
- Square footage (`sqft`) has the highest permutation importance, followed by `age_years` and `neighborhood_enc`.
- The image branch effectively learns price‑correlated visual patterns (e.g., richer colors for expensive houses, pool presence).
- Residual analysis indicates unbiased predictions across price ranges, though higher variance is observed for the most expensive properties.

**Conclusion**: This work demonstrates that a simple fusion of deep visual features with traditional tabular data can significantly improve housing price estimation, achieving an **MAE of ~$77K** and an **R² of 0.887** on a synthetic dataset designed to mimic real‑world complexity.