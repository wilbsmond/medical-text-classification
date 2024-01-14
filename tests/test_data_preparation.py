import pytest
from data_preparation import *
import os

# Fixtures for loaded & cleaned data
@pytest.fixture(scope="module")
def loaded_data():
    return load_data()

@pytest.fixture(scope="module")
def cleaned_data(loaded_data):
    df_user_inputs, df_labels = loaded_data
    return clean_data(df_user_inputs, df_labels)

# Test cases
def test_load_data(loaded_data):
    df_user_inputs, df_labels = loaded_data
    assert not df_user_inputs.empty, "User inputs dataframe is empty"
    assert not df_labels.empty, "Labels dataframe is empty"
    assert len(df_labels) == len(df_user_inputs), "Datasets do not align!"

def test_clean_data(cleaned_data):
    cleaned_user_inputs, cleaned_labels = cleaned_data
    assert 'unnecessary_column' not in cleaned_user_inputs.columns, "Unnecessary column not removed"
    assert cleaned_labels.shape[1] >= 2, "Classes with less than two instances not removed"

def test_preprocess_text():
    sample_text = "Example text w/ SPECIAL characters and 12345 numbers"
    expected_output = "example text w special characters and 12345 numbers"
    assert preprocess_text(sample_text) == expected_output, "Text preprocessing failed"

def test_split_train_test(cleaned_data):
    df_user_inputs, df_labels = cleaned_data
    train_texts, y_train, val_texts, y_val, test_texts, y_test = split_train_test(df_user_inputs, df_labels)

    assert train_texts.shape[0] == y_train.shape[0], "Mismatch in train data and labels"
    assert val_texts.shape[0] == y_val.shape[0], "Mismatch in train data and labels"
    assert test_texts.shape[0] == y_test.shape[0], "Mismatch in test data and labels"

    # Test the split ratio
    total_samples = len(df_user_inputs)
    assert train_texts.shape[0] / total_samples == pytest.approx(0.7, rel=0.05), "Train set size not as expected"
    assert val_texts.shape[0] / total_samples == pytest.approx(0.15, rel=0.05), "Validation set size not as expected"
    assert test_texts.shape[0] / total_samples == pytest.approx(0.15, rel=0.05), "Test set size not as expected"
