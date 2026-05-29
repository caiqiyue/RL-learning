import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import deepspeed


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO Training Script")
    parser.add_argument(
        "--model_name_or_path", type=str, default="gpt2", help="Model name or path"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="wikitext", help="Dataset name"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--deepspeed", type=str, default="ds_config.json", help="DeepSpeed config path"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    args = parser.parse_args()
    return args


def get_dataset(dataset_name, dataset_config):
    raw_datasets = load_dataset(dataset_name, dataset_config)
    if "validation" not in raw_datasets:
        raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=42)
        return raw_datasets["train"], raw_datasets["test"]
    return raw_datasets["train"], raw_datasets["validation"]


def preprocess_function(examples, tokenizer, max_length):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    args = parse_args()

    deepspeed.init_distributed()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto" if args.local_rank <= 0 else None,
    )

    train_dataset, eval_dataset = get_dataset(args.dataset_name, args.dataset_config)

    def tokenize(examples):
        return preprocess_function(examples, tokenizer, args.max_seq_length)

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    eval_dataset = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
