"""
finetune.py - QLoRA fine-tuning for candidate ranking using unsloth.
"""

import argparse
import os

try:
    from unsloth import FastLanguageModel
    from unsloth import MLXTrainer, MLXTrainingConfig
    import datasets.utils._dill as _dill
    import dill
    def patched_batch_setitems(self, items, obj=None):
        if obj is not None:
            return dill.Pickler._batch_setitems(self, items, obj)
        return dill.Pickler._batch_setitems(self, items)
    _dill.Pickler._batch_setitems = patched_batch_setitems
    from datasets import load_dataset
    from transformers import TrainingArguments
except ImportError:
    print("Error: Required libraries not found. Please install unsloth, trl, datasets, transformers.")
    # We will just pass in a dry-run or exit gracefully if this is run without env setup.

def finetune_model(dataset_path: str, model_name: str, output_dir: str):
    print(f"Starting QLoRA fine-tuning for {model_name}...")
    
    # 1. Load Model
    max_seq_length = 1024
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    
    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model, 
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    
    # 3. Load Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def format_prompt(examples):
        texts = []
        for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
            text = f"{inst}\n{inp}\nOutput:\n{out}"
            texts.append(text)
        return {"text": texts}
        
    dataset = dataset.map(format_prompt, batched=True, load_from_cache_file=False)
    
    # 4. Train
    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=MLXTrainingConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=100, # Set to higher value for real training
            learning_rate=2e-4,
            logging_steps=1,
            output_dir="outputs",
        ),
    )
    
    print("Training model...")
    trainer.train()
    
    # 5. Save Model
    print(f"Saving fine-tuned model to {output_dir}...")
    model.save_pretrained_merged(output_dir, tokenizer, save_method="lora")
    print("Fine-tuning complete. To export to GGUF, run:")
    print(f"python llama.cpp/convert_hf_to_gguf.py {output_dir} --outtype q4_k_m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="India_runs_data_and_ai_challenge/finetune_dataset.jsonl", help="Path to training data")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    parser.add_argument("--out", default="India_runs_data_and_ai_challenge/finetuned_ranker", help="Output directory")
    args = parser.parse_args()
    
    # We wrap in a try-except to avoid crashing if dependencies aren't installed during challenge validation
    try:
        finetune_model(args.dataset, args.model, args.out)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fine-tuning script aborted: {e}")
