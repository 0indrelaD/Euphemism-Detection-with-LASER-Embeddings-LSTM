# Euphemism Detection with LASER Embeddings + LSTM

A binary text classification project that detects euphemisms by combining multilingual **LASER sentence embeddings** with a custom **LSTM classifier**, using the sentence context alongside the Potentially Euphemistic Term (PET) itself.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model](#model)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Training Setup](#training-setup)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [Related Publication](#related-publication)
- [License](#license)

---

## 🩺 Overview

This project is an alternative approach to euphemism detection that, instead of fine-tuning a transformer end-to-end, uses **pretrained LASER sentence embeddings** (Language-Agnostic SEntence Representations) as fixed input features to a lightweight **LSTM classifier**. Two embeddings are computed per example — one for the full sentence (`text`) and one for the specific candidate phrase (`PET`, the Potentially Euphemistic Term) — and concatenated before being passed to the model.

This embedding-based approach is significantly cheaper to train than fine-tuning a full transformer, while still leveraging strong pretrained sentence representations, making it a useful lighter-weight baseline or ensemble component alongside transformer-based models.

---

## 📊 Dataset

The project uses the same style of English-language euphemism classification data as the transformer-based version of this task:

| File | Purpose |
|---|---|
| `EN_train.csv` | Training data with `text`, `PET`, and `label` columns |
| `EN_test.csv` | Unlabeled test data (text + PET only) |
| `EN_reference_test.csv` | Ground-truth labels for the test set |

Each row contains:
- `text` — the full sentence
- `PET` — the specific phrase being evaluated as potentially euphemistic
- `label` — binary label (Euphemism / Non-Euphemism), present only in the training data

---

## 🔄 Pipeline

### 1. Embedding Generation
- Both the `text` and `PET` columns are embedded separately using **LASER** (`laserembeddings`), a pretrained multilingual sentence embedding model
- The two embeddings are concatenated into a single feature vector per example, giving the model both broader sentence context and a focused representation of the candidate phrase

### 2. Train/Validation Split
- The embedded training data is split 80/20 into training and validation sets, stratified by label

### 3. Custom Dataset & DataLoader
- A `EuphemismDataset` class wraps the embeddings and labels as PyTorch tensors
- A `WeightedRandomSampler` is used during training to oversample the minority class on the fly

### 4. Model Training
- A custom LSTM-based classifier is trained with mixed-precision (`torch.cuda.amp`) for efficiency
- Training uses class-weighted loss, gradient clipping, a learning-rate scheduler, and early stopping based on validation F1

### 5. Evaluation
- Final validation and test-set performance are reported via classification reports and confusion matrices

---

## 🤖 Model

A custom single-directional **LSTM classifier** built in PyTorch:

- **Input:** Concatenated LASER embeddings (text + PET), treated as a single-timestep sequence
- **LSTM layer:** 2 layers, 256 hidden units, dropout 0.5
- **Head:** Fully connected layer (256 → 256) with ReLU and dropout, followed by a final linear layer producing a single logit
- **Output:** A single logit per example, passed through a sigmoid at inference time for binary classification (Euphemism vs. Non-Euphemism)

---

## ⚖️ Handling Class Imbalance

Since euphemism datasets tend to be imbalanced (non-euphemistic examples are typically far more common), this project addresses the imbalance in two complementary ways:

1. **Weighted sampling** — a `WeightedRandomSampler` oversamples minority-class examples during each training epoch
2. **Weighted loss** — `BCEWithLogitsLoss` is configured with a `pos_weight` derived from the computed class weights, penalizing misclassification of the minority class more heavily

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|---|---|
| Hidden dimension | 256 |
| LSTM layers | 2 |
| Dropout | 0.5 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 2) |
| Max epochs | 30 |
| Early stopping patience | 7 epochs (active after epoch 10) |
| Mixed precision | Enabled (`torch.cuda.amp`) |
| Gradient clipping | Max norm 1.0 |
| Model selection metric | Validation F1 (best checkpoint saved) |

---

## 📈 Evaluation

- **Per-epoch metrics:** training/validation loss, accuracy, and F1-score are printed each epoch, along with a full classification report on the validation set
- **Final validation evaluation:** classification report and confusion matrix using the best-F1 checkpoint
- **Test set evaluation:** classification report and confusion matrix against the independent reference labels in `EN_reference_test.csv`

---

## 🧰 Tech Stack

- **Language:** Python 3
- **Deep Learning:** PyTorch (custom LSTM, mixed-precision training)
- **Embeddings:** `laserembeddings` (LASER multilingual sentence embeddings)
- **Data Handling:** Pandas, NumPy
- **ML Utilities:** Scikit-learn (splitting, class weights, metrics)
- **Visualization:** Matplotlib
- **Environment:** Google Colab (GPU recommended)

---

## 📁 Project Structure

```
euphemism-lstm-laser/
│
├── euphemism_lstm.ipynb        # Main notebook: embeddings, model, training, evaluation
├── data/
│   ├── EN_train.csv            # Training data (text, PET, label)
│   ├── EN_test.csv             # Test data (text, PET)
│   └── EN_reference_test.csv   # Ground-truth test labels
├── best_model_f1.pth           # Best checkpoint by validation F1
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
sudo apt-get install -y cmake build-essential python3-dev
pip install torch torchvision torchaudio transformers datasets \
            scikit-learn pandas numpy tqdm sentencepiece \
            sacremoses subword-nmt laserembeddings==1.1.2
python -m laserembeddings download-models
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/euphemism-lstm-laser.git
   cd euphemism-lstm-laser
   ```
2. Place `EN_train.csv`, `EN_test.csv`, and `EN_reference_test.csv` in the `data/` folder, updating the file paths in the notebook (the original paths point to Google Drive).
3. Open and run the notebook:
   ```bash
   jupyter notebook euphemism_lstm.ipynb
   ```

> **Note:** LASER model downloads and embedding generation can take a few minutes on first run. A GPU is recommended for training, though the LASER embedding step itself runs on CPU.

---

## 💡 Key Insights

- Concatenating a **candidate-phrase embedding** (PET) alongside the **full-sentence embedding** gives the model both local and contextual signal — useful since whether a phrase is euphemistic often depends on how it's used in context, not just the phrase itself.
- A **frozen pretrained embedding + lightweight classifier** approach trains much faster than full transformer fine-tuning, making it practical for quick iteration or as one component of an ensemble.
- Combining **weighted sampling and weighted loss** is a stronger imbalance-handling strategy than either alone, since it addresses the imbalance both at the data-loading level and the loss-function level.
- **Early stopping on F1** (rather than loss or accuracy) is a better choice for imbalanced binary classification, since F1 accounts for both precision and recall on the positive class.

---

## 🔮 Future Improvements

- Fix the checkpoint filename mismatch: the model is saved as `best_model_f1.pth` during training but loaded as `best_model.pth` in the final evaluation step — this will currently raise a file-not-found error and should be corrected before running end-to-end.
- Experiment with a **bidirectional LSTM** to let the model attend to embedding information from both directions, though this matters less given the single-timestep embedding input.
- Try alternative fixed embeddings (e.g. Sentence-BERT, LaBSE) as a comparison point against LASER.
- Combine this LSTM-based model with the transformer-based approach (see the related XLNet euphemism detection project) in a **blending ensemble**, consistent with the Blend-XED research this work connects to.
- Add a validation-set ROC/AUC curve for a threshold-independent view of performance, complementing the classification report.

---

## 📄 Related Publication

This project's embedding-based approach is one of the components explored in ensemble-based euphemism detection research:

**Blend-XED: A Transformer-Based Blending Ensemble Model for Euphemism Detection**
Oindrela, D., Faiza, S. R., Islam, T., Chy, A. N., & Ahmed, T. — Accepted and published at the IEEE International Black Sea Conference on Communications and Networking (BlackSeaCom), Bucharest, Romania, June 2026.

---

## 📄 License

This project is open-source and available for educational and research purposes. Please cite appropriately if reused.
