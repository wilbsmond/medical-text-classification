from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model_building_trf import tokenize_and_aggregate

# Load saved model and tokenizer
# Load the model and tokenizer
def load_checkpoint_model_and_tokenizer(saved_path="models/trf"):
    model = AutoModelForSequenceClassification.from_pretrained(f"{saved_path}/model_trf")
    tokenizer = AutoTokenizer.from_pretrained(f"{saved_path}/tokenizer_trf")
    return model, tokenizer

def classify_text(model, user_text):
    # How to do inference on new text (after tokenization)?
    results = model.predict?(user_text)

if __name__ == "__main__":
    model, tokenizer = load_checkpoint_model_and_tokenizer(saved_path="models/trf")
    tokenized_user_text = tokenize_and_aggregate(tokenizer, user_text)

