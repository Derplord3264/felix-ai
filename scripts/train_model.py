from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd
import torch

def train_model():
    # Load dataset
    df = pd.read_csv('data/dataset.csv')
    conversations = df['conversation'].tolist()

    # Tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Tokenize data
    inputs = tokenizer(conversations, return_tensors='pt', padding=True, truncation=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs,
    )

    # Train model
    trainer.train()
    model.save_pretrained('model')
    tokenizer.save_pretrained('model')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
