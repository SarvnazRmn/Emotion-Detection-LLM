# 🔥 Best Performing Model: LLaMA with Layer-Specific Learning Rates

## 🧠 Model Overview

This notebook implements a fine-tuned LLaMA-based architecture for multilabel emotion classification using the BRIGHTER dataset (English track A) from SemEval2025 Task 11.

The model is built on top of the pretrained `meta-llama/Llama-3.2-1B` and features a custom training strategy that selectively unfreezes deeper transformer layers while applying **layer-specific learning rates**. This enabled stable and effective fine-tuning while preventing the explosion in validation loss.

### ❗ Problem

Initially, unfreezing the last four layers of the LLaMA backbone caused **exploding validation loss**, due to excessive parameter updates in deeper parts of the network.

### ✅ Solution

We addressed this by introducing **differentiated learning rates**:
- Classification head: `1e-5`
- Unfrozen LLaMA layers (12–15): `1e-6`  
This tenfold reduction in learning rate for the base model layers stabilized training.

## ⚙️ Architecture Summary

- **Backbone**: `meta-llama/Llama-3.2-1B`
- **Fine-tuning Strategy**:
  - Freeze all LLaMA layers initially
  - Unfreeze layers `12`, `13`, `14`, `15`
  - Apply custom optimizer with layer-specific learning rates
- **Classifier Head**: Linear MLP layers added for emotion classification
- **Loss Function**: `BCEWithLogitsLoss` (multi-label classification)
- **Max Token Length**: 128
- **Evaluation Metrics**: Micro and Macro F1 scores
- **Training Framework**: Hugging Face `Trainer` API
- **Training Environment**: Google Colab with GPU

## 📈 Final Evaluation Results (English Track A)

**Micro-Averaged Scores**
- Precision: `0.7834`
- Recall: `0.7187`
- F1 Score: `0.7496`

**Macro-Averaged Scores**
- Precision: `0.7689`
- Recall: `0.6697`
- F1 Score: `0.7114`

## 🏆 Leaderboard Comparison

Our best model's macro F1 score of **`0.7114`** surpasses the **top-ranked system** in the official SemEval2025 leaderboard, which achieved a macro F1 of **`0.6986`**.

🔗 [Official Leaderboard](https://docs.google.com/spreadsheets/d/1pyKXpvmVsE1rwu8aRnfn372rD-v3L4Qo/edit?gid=57247028#gid=57247028)

## 📁 Files in This Folder

- `LLAMA_4layer_unfreezed.ipynb`: Full training and evaluation pipeline
- `pred_eng.csv`: Final model predictions on the test set
- `evaulation.ipynb`: Evalution of the Predictions and check it to be in the right format of the submission for SemEval2025-Task11

---

📌 *This directory represents the best performing configuration in our submission to SemEval2025 Task 11.*
