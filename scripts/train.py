from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk

def train_model():
    # Load processed dataset
    dataset = load_from_disk("data/processed/tokenized_dataset")
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained("nisten/Biggie-SmoLlm-0.15B-Base")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )
    
    # Train model
    trainer.train()
