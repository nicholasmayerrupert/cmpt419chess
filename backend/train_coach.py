import os

# -------- GLOBAL ENV / COMPILATION SWITCHES --------
os.environ["WANDB_DISABLED"] = "true"

# Use 12 processes for datasets/Unsloth
os.environ["HF_DATASETS_NUM_PROC"] = "12"

# Keep BLAS from spawning tons of threads per process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 🔴 Turn OFF Unsloth’s auto-compiler
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # docs-unsloth

# 🔴 Turn OFF PyTorch compile / Inductor
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
# ---------------------------------------------------

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


def main():
    max_seq_length = 1024
    dtype = None
    load_in_4bit = True

    print("Loading Llama 3.1 8B (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    alpaca_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for inp, out in zip(inputs, outputs):
            texts.append(alpaca_prompt.format(inp, out))
        return {"text": texts}

    print("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files="chess_coaching_data.jsonl",
        split="train",
    )

    # 12 workers for tokenization
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=12,
    )

    print("Starting Training...")
    training_args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",

        dataloader_num_workers = 0,
        dataloader_pin_memory = False,

        dataset_num_proc = 12,
        torch_compile = False,   # belt + suspenders
        report_to = "none",
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        packing = False,
        args = training_args,
    )

    trainer.train()

    print("Saving GGUF...")
    model.save_pretrained_gguf(
        "model_checkpoints",
        tokenizer,
        quantization_method = "q4_k_m",
    )
    print("Done! Your model is in 'model_checkpoints/unsloth.Q4_K_M.gguf'")


if __name__ == "__main__":
    main()
