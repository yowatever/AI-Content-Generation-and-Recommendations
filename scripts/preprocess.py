from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_data(dataset_name="wikitext", subset="wikitext-2-raw-v1"):
    """Load and preprocess the dataset"""
    tokenizer = AutoTokenizer.from_pretrained("nisten/Biggie-SmoLlm-0.15B-Base")
    dataset = load_dataset(dataset_name, subset)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Save processed data
    tokenized_dataset.save_to_disk("data/processed/tokenized_dataset")
    
    return tokenized_dataset
