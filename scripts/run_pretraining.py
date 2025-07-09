import os

import torch
from datasets import load_dataset, IterableDataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from src.t5_rope_model import T5Config, T5ForConditionalGeneration, chunk_and_tokenize_stream, \
    preprocess_for_t5_denoising


def main():
    os.environ["WANDB_PROJECT"] = "c4-rope-t5-pretraining-stream"

    tokenizer_name = "t5-small"
    model_output_dir = "./c4-rope-t5-from-scratch-stream"

    print(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Initializing a new T5 model from scratch...")
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        _attn_implementation="flash_attention_2",
    )

    model = T5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    print(f"Model created with {model.num_parameters():,} parameters.")

    print("Loading C4 dataset in STREAMING mode...")
    full_dataset = load_dataset("allenai/c4", "en", streaming=True)
    train_stream = full_dataset['train']

    shuffled_stream = train_stream.shuffle(seed=42, buffer_size=10_000)

    print("Applying transformations (chunking, tokenizing, and denoising) on the fly...")
    chunked_tokenized_stream = chunk_and_tokenize_stream(shuffled_stream, tokenizer, chunk_size=512)

    processed_stream_generator = IterableDataset.from_generator(
        lambda: chunked_tokenized_stream
    )

    denoised_stream = processed_stream_generator.map(
        lambda examples: preprocess_for_t5_denoising(examples, tokenizer),
        batched=True,
        batch_size=256,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        max_steps=1_000_000,
        bf16=True,
        torch_compile=False,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=2,
        logging_steps=500,
        save_steps=10000,
        save_total_limit=3,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        weight_decay=0.01,
        dataloader_num_workers=4,
        report_to=["wandb"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=denoised_stream,
        data_collator=data_collator,
    )

    print("🚀 Starting distributed training on the streaming C4 dataset...")
    trainer.train()
    print("✅ Training complete.")

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")


if __name__ == "__main__":
    main()