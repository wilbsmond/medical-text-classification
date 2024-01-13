import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import f1_score,  precision_score, recall_score, hamming_loss

## ============== 1. Load data ===============================
# Load the cleaned datasets
df_user_inputs = pd.read_csv('dataset/user_inputs_cleaned.csv')
df_labels = pd.read_csv('dataset//labels_cleaned.csv')

# Remove unnecessary index columns
df_user_inputs.drop(df_user_inputs.columns[0], axis=1, inplace=True)
df_labels.drop(df_labels.columns[0], axis=1, inplace=True)

# Ensure alignment
assert len(df_labels) == len(df_user_inputs), "Datasets do not align!"

## ============== 2. Prepare data to model format ================
## Split data to train:val:test

# Prepare data for iterative train test split
# X must be 2D np.ndarray and y must be 2D binary np.ndarray
X_texts = df_user_inputs['text'].values
X_texts = X_texts.reshape(-1, 1)
y = df_labels.values

# Split the data 60:20:20 with multi-label stratification
train_texts, y_train, test_texts, y_test = iterative_train_test_split(X_texts, y, test_size = 0.2)
#val_texts, y_val, test_texts, y_test = iterative_train_test_split(tmp_texts, y_tmp, test_size = 0.5)

# Sanity checks to confirm the shapes of the datasets
assert train_texts.shape[0] == y_train.shape[0], "Mismatch in train data and labels"
assert test_texts.shape[0] == y_test.shape[0], "Mismatch in test data and labels"

train_texts, test_texts = train_texts.ravel(), test_texts.ravel()
#val_texts = val_texts.ravel()

