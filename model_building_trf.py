import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score,  precision_score, recall_score, hamming_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import evaluate

def load_prepped_data(data_path="./dataset"):
    # Load the datasets
    df_train_texts = pd.read_csv(f'{data_path}/train_texts.csv', delimiter=';')
    df_val_texts = pd.read_csv(f'{data_path}/val_texts.csv', delimiter=";")
    df_test_texts = pd.read_csv(f'{data_path}/test_texts.csv', delimiter=";")
    df_y_train = pd.read_csv(f'{data_path}/y_train.csv', delimiter=';')
    df_y_val = pd.read_csv(f'{data_path}/y_val.csv', delimiter=";")
    df_y_test = pd.read_csv(f'{data_path}/y_test.csv', delimiter=";")

    train_texts = df_train_texts['text'].values
    val_texts = df_val_texts['text'].values
    test_texts = df_test_texts['text'].values

    y_train = df_y_train.values
    y_val = df_y_val.values
    y_test = df_y_test.values

    return train_texts, y_train, val_texts, y_val, test_texts, y_test

def load_pretrained_tokenizer_and_model(model_name="GroNLP/bert-base-dutch-cased"):
    # Use GPU if available
    if torch.cuda.is_available():
        print("CUDA is available. You can use GPU.")
    else:
        print("CUDA is not available. Check your GPU setup.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=y_train.shape[1],
                                                               problem_type="multi_label_classification")
    model.to(device)

    return tokenizer, model

# Tokenize each text individually and aggregate the results
def tokenize_and_aggregate(tokenizer, texts):
    tokenized_texts = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    for text in texts:
        tokenized_text = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        tokenized_texts['input_ids'].append(tokenized_text['input_ids'][0])
        tokenized_texts['attention_mask'].append(tokenized_text['attention_mask'][0])
        if 'token_type_ids' in tokenized_text:
            tokenized_texts['token_type_ids'].append(tokenized_text['token_type_ids'][0])
    return tokenized_texts

# Prepare dataset for TRF format
class PrepareDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Apply sigmoid to logits
    predictions = 1 / (1 + np.exp(-logits))
    # Convert to binary values (0 or 1) with a threshold, e.g., 0.5
    threshold = 0.5
    predictions = (predictions > threshold).astype(int)

    # Compute metrics for each label and then average (micro)
    f1_macro = f1_score(labels, predictions, average='micro')
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    hamming_loss_value = hamming_loss(labels, predictions)

    return {
        'f1': f1_macro,
        'precision': precision,
        'recall': recall,
        'hamming_loss': hamming_loss_value,
    }

def train_model(model, train_dataset, val_dataset, compute_metrics):
    # Training arguments
    training_args = TrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        evaluation_strategy="epoch",
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    return model, trainer

def evaluate_model(trainer, test_dataset):
    # Evaluate the model
    results = trainer.evaluate(test_dataset)

    # Filter the dictionary to include only the desired metrics
    filtered_results = {key: results[key] for key in results if key in ['eval_loss', 'eval_f1_micro', 'eval_precision', 'eval_recall', 'eval_hamming_loss']}

    # Convert to DataFrame and Transpose it
    results_df = pd.DataFrame([filtered_results]).T
    results_df.columns = ['Value']  # You can rename the column header as needed

    return results_df

def save_model_and_tokenizer(model, tokenizer, save_path="models/trf"):
    # Save the model
    model_save_path = f"{save_path}/model_trf"
    model.save_pretrained(model_save_path)

    # Save the tokenizer in the same way, if we need it later
    tokenizer_save_path = f"{save_path}/tokenizer_trf"
    tokenizer.save_pretrained(tokenizer_save_path)

if __name__ == "__main__":
    train_texts, y_train, val_texts, y_val, test_texts, y_test = load_prepped_data("./dataset")
    tokenizer, model = load_pretrained_tokenizer_and_model("GroNLP/bert-base-dutch-cased")

    # Tokenize texts
    tokenized_train_data = tokenize_and_aggregate(tokenizer, train_texts)
    tokenized_val_texts = tokenize_and_aggregate(tokenizer, val_texts)
    tokenized_test_texts = tokenize_and_aggregate(tokenizer, test_texts)

    # Prepare our data for TRF format
    train_dataset = PrepareDataset(tokenized_train_data, y_train)
    val_dataset = PrepareDataset(tokenized_val_texts, y_val)
    test_dataset = PrepareDataset(tokenized_test_texts, y_test)

    # Train model
    trainer = train_model(model, train_dataset, val_dataset, compute_metrics)

    # Evaluate model
    results_df = evaluate_model(trainer, test_dataset)
    print(results_df)

    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer, "models/trf")