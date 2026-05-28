import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def build_examples(df, prompt_prefix="\nPoem:\n", tone_prefix="\n\nTone:"):
    texts = []
    for poem, tone in zip(df["Themes"].fillna(""), df["Tone"].fillna("")):
        inp = f"{prompt_prefix}{poem}{tone_prefix} {tone}"
        texts.append({"text": inp})
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="sarojini_naidu_poetry_dataset.csv")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b", help="HF model id or local path")
    parser.add_argument("--output_dir", default="llama_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    examples = build_examples(df)
    ds = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch):
        toks = tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        toks["labels"] = toks["input_ids"].copy()
        return toks

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.use_lora:
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
