import re
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

## ---------- Load data -------------------------------
# Load the datasets
def load_data():
    df_user_inputs = pd.read_csv('dataset/user_inputs.csv', delimiter=';')
    df_labels = pd.read_csv('dataset/labels.csv', delimiter=";")
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

# Load the Dutch language model from Spacy
#nlp = spacy.load("nl_core_news_sm")

# Set of Dutch stopwords from NLTK
dutch_stopwords = set(stopwords.words('dutch'))

def preprocess_text(text):
    """
    Preprocesses the input text by lowercasing, removing special characters, and removing stopwords.
    Args:
        text (str): The text to preprocess.
    Returns:
        str: The preprocessed text.
    """

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

def save_to_csv(df_user_inputs, df_labels):
    df_user_inputs.to_csv('dataset/user_inputs_cleaned.csv')
    df_labels.to_csv('dataset/labels_cleaned.csv')

## Main
df_user_inputs, df_labels = load_data()
df_user_inputs, df_labels = clean_data(df_user_inputs, df_labels)
assert len(df_labels) == len(df_user_inputs), "Datasets do not align!"

df_user_inputs['text'] = df_user_inputs['text'].apply(preprocess_text)

save_to_csv(df_user_inputs, df_labels)