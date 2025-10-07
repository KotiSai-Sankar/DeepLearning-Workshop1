# Binary Classification with Neural Networks on the Census Income Dataset
This project builds a binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually, based on the Census Income Dataset (income.csv) containing 30,000 entries.
## Done By:
 Name : Koti Sai Sankar  
 Rgister Number : 212222240111
## Task Description
Dataset: income.csv (30,000 entries).

## Goal: Predict income level (<=50K or >50K).

## Data Preparation
### Separate columns:
Categorical features

Continuous features

Label column (income).

### Transform data:
Convert categorical values to category codes.

Standardize continuous features.

Encode labels (0 = <=50K, 1 = >50K).

### Create tensors:
Categorical values → LongTensors.

Continuous values → FloatTensors.

Labels → LongTensors.

### Split dataset:
-- Training set: 25,000 samples.

-- Test set: 5,000 samples.

## Model Design
### TabularModel Class:
Embeddings for categorical features.

BatchNorm for continuous features.

One hidden layer with 50 neurons.

Dropout (p=0.4) for regularization.

Output layer with 2 classes (<=50K, >50K).

## Training
Random seed set for reproducibility.

Loss function: CrossEntropyLoss.

Optimizer: Adam (lr=0.001).

Epochs: 300.

Training loop: forward pass → loss calculation → backward pass → optimizer step.

## Evaluation
Evaluate on the test set (5,000 samples).

Report:

Test Loss

Accuracy (%)

## How to Run
1) Install dependencies:

       ```pip install torch pandas scikit-learn```
2) Place income.csv in the working directory.

3) Run training script:

       ```python train.py```
4) After training, the script prints test loss and accuracy.

5) Final accuracy ~ 80–85% (depending on random seed and preprocessing).
