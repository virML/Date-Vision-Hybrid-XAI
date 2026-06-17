# Date-Vision-Hybrid: Dual-Stream Framework for Date Fruit Classification

**Recommended Repository Name:** `Date-Vision-Hybrid-XAI`

## Description
This repository contains the implementation of a novel **Dual-Stream Hybrid Framework** designed for the automated classification of Saudi Arabian date fruits. The project addresses the challenges of inter-class similarity in agriculture by synergistically combining **Convolutional Neural Networks (CNNs)** for local texture extraction and **Vision Transformers (ViTs)** for global context modeling. The final classification is performed using an optimized **XGBoost** head, with model transparency provided through **Explainable AI (SHAP)**.

## Dataset Information
The models are trained on the **Date Fruit Image Dataset in Controlled Environment**, which is publicly available on Kaggle.
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment)
* **Content:** 1,658 high-quality JPG images.
* **Classes (9):** Ajwa, Galaxy, Medjool, Meneifi, Nabtat Ali, Rutab, Shaishe, Sokari, and Sugaey.
* **Setup:** Images were captured using a DSLR (Canon EOS 550D) at a fixed distance of 8 cm with a 240 LED ring light to eliminate shadows and emphasize surface "fleshiness."

## Code Information
The repository includes three primary implementation notebooks, each exploring a different point on the accuracy-latency spectrum:
1.  **`swin_base_n_levit_256_fixed.ipynb`**: The flagship model using **Swin-Base** and **LeViT-256**. Offers the highest accuracy (96.79%) for high-performance requirements.
2.  **`swin_tiny_n_levit_128_fixed.ipynb`**: A balanced approach using **Swin-Tiny** and **LeViT-128s**, optimizing for efficiency without significant loss in precision.
3.  **`efficient_n_levit_128_fixed.ipynb`**: A hybrid stream utilizing **EfficientNetV2-M** and **LeViT-128s**, focused on minimized computational footprint.

## Methodology
The framework follows a systematic pipeline:
1.  **Dual-Stream Extraction:** Images are processed simultaneously through a CNN/Swin-Transformer stream (local features) and a LeViT stream (global dependencies).
2.  **Feature Fusion:** Features from both streams are concatenated and refined through **Global Average Pooling (GAP)**.
3.  **XGBoost Classification:** Instead of a standard Softmax layer, the fused features are passed to an **XGBoost** classifier tuned via **Optuna** (TPE sampler).
4.  **Explainability:** **SHAP (SHapley Additive exPlanations)** is applied to generate heatmaps, ensuring the model focuses on biologically relevant features of the fruit rather than background noise.

## Requirements
To run these notebooks, you will need:
* **Python:** 3.9+
* **Core Libraries:** `torch`, `torchvision`, `timm` (for Swin and LeViT architectures)
* **Machine Learning:** `xgboost`, `optuna`, `scikit-learn`
* **Explainability:** `shap`
* **Data Handling:** `numpy`, `pandas`, `matplotlib`, `PIL`

## Usage Instructions
1.  **Environment Setup:**
    ```bash
    pip install torch torchvision timm xgboost optuna shap
    ```
2.  **Data Preparation:** Download the dataset from Kaggle and organize it into class-specific folders. Update the `data_dir` path in the notebooks.
3.  **Training:** Execute the notebooks sequentially. Each notebook includes:
    * Hyperparameter tuning with Optuna.
    * Feature extraction and XGBoost training.
    * Evaluation metrics (Confusion Matrix, F1-Score, AUROC).
4.  **Inference:** Use the saved model weights to classify new date images via the provided `predict` functions.

## Citations
If you utilize this code or the associated methodology in your research, please cite:

## License & Contributions
This project is licensed under the MIT License. Contributions are welcome—please feel free to submit a Pull Request or open an issue for discussion.
