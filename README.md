# LELA60342: Sentiment Classification of Amazon Reviews

This repository contains the code, submission scripts, and output logs for the RM2 programming assignment, training neural networks to classify sentiment in Amazon reviews.

## Repository Contents
* `RM2Proj.py`: The complete Python script encompassing data loading, preprocessing, PyTorch model definitions, training loops, and statistical evaluation.
* `run_model.slurm`: The Slurm batch script used to execute the code on the CSF3 A100 GPU nodes.
* `slurm-[YOUR_JOB_ID].out`: The raw output log from the cluster showing the training progress, final AUC scores, and the bootstrapped p-value.

---

## Description and Discussion of Models (2 Marks)

The data was preprocessed using a `TfidfVectorizer` (max 5,000 features, unigrams/bigrams, `min_df=3` to remove noisy/rare words). Two models were then implemented using PyTorch:

**Model 1: Baseline Logistic Regression**
The baseline model is a simple logistic regression classifier implemented as a single linear layer (`nn.Linear`) followed by a Sigmoid activation. It was trained using standard Binary Cross Entropy Loss (`BCELoss`) and the AdamW optimizer. 

**Model 2: Improved Deep Deep Learning Classifier (MLP)**
To improve upon the linear baseline, Model 2 implements a deep Multi-Layer Perceptron (MLP) designed to capture non-linear relationships while strictly controlling for overfitting and dataset biases. Key additions include:
1. **Architecture:** Three hidden layers (512 $\rightarrow$ 256 $\rightarrow$ 64) allowing the network to learn complex feature representations.
2. **Regularization:** Each layer incorporates Batch Normalization (`BatchNorm1d`), Leaky ReLUs to prevent dead neurons, and a high Dropout rate (0.3) to force the network to generalise rather than memorise the training data.
3. **Class Imbalance Handling:** The dataset is skewed toward positive reviews. Model 2 swaps `BCELoss` for `BCEWithLogitsLoss` using a calculated `pos_weight` (0.7420). This penalises the model equally for minority and majority class errors.
4. **Dynamic Training:** The model uses a learning rate scheduler (`ReduceLROnPlateau`) and Early Stopping (patience=25) based on a 10% validation holdout set to halt training exactly when the model stops generalising.

---

## Description and Discussion of Results 

The models were evaluated on a 20% test holdout set using the Area Under the Receiver Operating Characteristic Curve (AUC).

* **Model 1 AUC:** 0.8417
* **Model 2 AUC:** 0.9255
* **Improvement:** +0.0838
* **Bootstrapped p-value:** $p < 0.001$ (1,000 iterations)

**Discussion:**
Model 2 significantly outperformed the baseline logistic regression ($p < 0.001$). The preprocessing (`min_df=3`) successfully prevented Model 1 from artificially inflating its score by memorising rare text artifacts, resulting in a true baseline AUC of 0.8417. 

By contrast, Model 2 achieved an excellent AUC of 0.9255. Because Model 2 was trained using `BCEWithLogitsLoss` with positive class weighting, this high AUC reflects genuine predictive power across *both* positive and negative reviews, rather than a biased reliance on guessing the majority class. Furthermore, the early stopping mechanism (halting at epoch 55) and high dropout rates ensured the model did not overfit, demonstrating that the deep MLP architecture effectively captured deep semantic patterns within the TF-IDF vectors that the linear baseline could not.
