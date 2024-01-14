import re
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
#import spacy
#import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')

## ---------- Load data -------------------------------
# Load the datasets
def load_data(data_path="dataset/raw_data"):
    df_user_inputs = pd.read_csv(f'{data_path}/user_inputs.csv', delimiter=';')
    df_labels = pd.read_csv(f'{data_path}/labels.csv', delimiter=";")
    return df_user_inputs, df_labels

def clean_data(df_user_inputs, df_labels):
    # Remove unnecessary index columns
    df_user_inputs.drop(df_user_inputs.columns[0], axis=1, inplace=True)
    df_labels.drop(df_labels.columns[0], axis=1, inplace=True)

    # Remove classes with < 2 instances (this is only 'no complaints' label with 0 instance so not a big deal)
    # We need to do this to split the data later with stratification
    df_labels = df_labels.loc[:, (df_labels.sum(axis=0) >= 2)]

    return df_user_inputs, df_labels

## ------------ Preprocess user input text -----------------------
def preprocess_text(text):
    """
    Preprocesses the input text by lowercasing, removing special characters, and removing stopwords.
    Args:
        text (str): The text to preprocess.
    Returns:
        str: The preprocessed text.
    """
    # Load the Dutch language model from Spacy
    #nlp = spacy.load("nl_core_news_sm")

    # Set of Dutch stopwords from NLTK
    #dutch_stopwords = set(stopwords.words('dutch'))

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'\W+', ' ', text)
    """
    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in dutch_stopwords]

    # Lemmatize each token
    doc = nlp(" ".join(filtered_tokens))
    lemmas = [token.lemma_ for token in doc]

    text = ' '.join(lemmas)
    """
    return text

def split_train_test(df_user_inputs, df_labels):
    # Prepare data for iterative train test split
    # X must be 2D np.ndarray and y must be 2D binary np.ndarray
    X_texts = df_user_inputs['text'].values
    X_texts = X_texts.reshape(-1, 1)
    y = df_labels.values

    # Split the data 70:15:15 with multi-label stratification
    SEED = 42
    np.random.seed(SEED)
    train_texts, y_train, tmp_texts, y_tmp = iterative_train_test_split(X_texts, y, test_size = 0.3)
    val_texts, y_val, test_texts, y_test = iterative_train_test_split(tmp_texts, y_tmp, test_size = 0.5)

    train_texts, val_texts, test_texts = train_texts.ravel(), val_texts.ravel(), test_texts.ravel()

    return train_texts, y_train, val_texts, y_val, test_texts, y_test

def save_to_csv(train_texts, y_train, val_texts, y_val, test_texts, y_test, df_labels):
    # Convert texts to DataFrame
    train_texts_df = pd.DataFrame(train_texts, columns=['text'])
    val_texts_df = pd.DataFrame(val_texts, columns=['text'])
    test_texts_df = pd.DataFrame(test_texts, columns=['text'])

    # Convert labels to DataFrame
    y_train_df = pd.DataFrame(y_train, columns=df_labels.columns)
    y_val_df = pd.DataFrame(y_val, columns=df_labels.columns)
    y_test_df = pd.DataFrame(y_test, columns=df_labels.columns)

    # Save texts and labels to separate CSV files
    train_texts_df.to_csv('dataset/train_texts.csv', index=False, sep=';')
    val_texts_df.to_csv('dataset/val_texts.csv', index=False, sep=';')
    test_texts_df.to_csv('dataset/test_texts.csv', index=False, sep=';')

    y_train_df.to_csv('dataset/y_train.csv', index=False, sep=';')
    y_val_df.to_csv('dataset/y_val.csv', index=False, sep=';')
    y_test_df.to_csv('dataset/y_test.csv', index=False, sep=';')

if __name__ == "__main__":
    df_user_inputs, df_labels = load_data("dataset/raw_data")
    df_user_inputs, df_labels = clean_data(df_user_inputs, df_labels)
    df_user_inputs['text'] = df_user_inputs['text'].apply(preprocess_text)
    train_texts, y_train, val_texts, y_val, test_texts, y_test = split_train_test(df_user_inputs, df_labels)
    save_to_csv(train_texts, y_train, val_texts, y_val, test_texts, y_test, df_labels)
