from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk

def train_model():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    dataset = load_from_disk('data/daily_dialog')
    tokenized_dataset = dataset.map(lambda x: tokenizer(x['dialog'], padding='max_length', truncation=True, max_length=512), batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation']
    )

    trainer.train()
    model.save_pretrained('model/t5-small')

if __name__ == "__main__":
    train_model()
