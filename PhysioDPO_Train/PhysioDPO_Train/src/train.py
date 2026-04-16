import os
import sys
import torch
import argparse
from typing import Any
import types
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trainer import PhysioDPOTrainer
from utils import PhysioDataCollator, find_all_linear_names


def _find_first_embedding_module(model: nn.Module) -> nn.Embedding:
    for _, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            return module
    raise RuntimeError("Could not find any nn.Embedding module in the loaded model.")


def patch_progen_input_embeddings(model: nn.Module) -> None:
    def _patch_one(m: nn.Module) -> None:
        try:
            _ = m.get_input_embeddings()  # type: ignore[attr-defined]
            return
        except NotImplementedError:
            pass
        except Exception:
            pass

        emb = _find_first_embedding_module(m)

        def _get_input_embeddings(self):  # type: ignore     getter
            return emb

        def _set_input_embeddings(self, new_embeddings):  # type: ignore     setter
            nonlocal emb
            if not isinstance(new_embeddings, nn.Embedding):
                raise TypeError("new_embeddings must be an nn.Embedding")
            emb = new_embeddings

        m.get_input_embeddings = types.MethodType(_get_input_embeddings, m)  # type: ignore
        m.set_input_embeddings = types.MethodType(_set_input_embeddings, m)  # type: ignore

    _patch_one(model)

    for attr in ["base_model", "model", "transformer"]:
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module):
            _patch_one(inner)


def disable_gradient_checkpointing_for_progen(model: nn.Module) -> None:

    def _no_gc(self, *args, **kwargs):  # type: ignore
        return None

    for m in [model, getattr(model, "model", None), getattr(model, "base_model", None)]:
        if not isinstance(m, nn.Module):
            continue
        if hasattr(m, "gradient_checkpointing_enable"):
            m.gradient_checkpointing_enable = types.MethodType(_no_gc, m)  # type: ignore
        if hasattr(m, "is_gradient_checkpointing"):
            try:
                setattr(m, "is_gradient_checkpointing", False)
            except Exception:
                pass
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "gradient_checkpointing"):
            try:
                cfg.gradient_checkpointing = False
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="PhysioDPO Training")
    parser.add_argument("--model_id", type=str, default="hugohrban/progen2-xlarge",
                        help="Base model ID")
    parser.add_argument("--data_path", type=str, default="data/demo.json",
                        help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/physio-dpo-progen2-xl-1m",
                        help="Output directory for checkpoints")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Micro batch size per device")
    parser.add_argument("--grad_accum_steps", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--mu", type=float, default=50.0,
                        help="Physio parameter mu")
    parser.add_argument("--tau", type=float, default=10.0,
                        help="Physio parameter tau")
    parser.add_argument("--lambda_param", type=float, default=1.0,
                        help="Physio parameter lambda")
    parser.add_argument("--single_gpu", action="store_true",
                        help="Use single GPU mode (no DeepSpeed)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    MODEL_ID = args.model_id
    DATA_PATH = args.data_path
    OUTPUT_DIR = args.output_dir
    
    MICRO_BATCH_SIZE = args.micro_batch_size
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    
    MAX_STEPS = args.max_steps
    LEARNING_RATE = args.learning_rate
    MAX_LENGTH = args.max_length
    
    PHYSIO_PARAMS = {
        "mu": args.mu,
        "tau": args.tau,
        "lambda": args.lambda_param
    }

    print(f"Loading dataset from {DATA_PATH}...")
    try:
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir="./cache")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading model from {MODEL_ID}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_cache=False,
            device_map="auto",
            cache_dir="./cache"
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying without flash attention...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_cache=False,
            device_map="auto",
            cache_dir="./cache"
        )
    

    patch_progen_input_embeddings(model)
    disable_gradient_checkpointing_for_progen(model)

    print("Model preparation complete")

    target_modules = find_all_linear_names(model)
    print(f"LoRA Target Modules: {target_modules}")
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    # Configure training arguments based on single_gpu flag
    common_args = {
        "output_dir": OUTPUT_DIR,
        "per_device_train_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 1000,
        "bf16": True,
        # ProGen does not support gradient checkpointing; keep it off to avoid PEFT errors.
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
        "run_name": "physio-dpo-final",
        "optim": "paged_adamw_32bit",
    }
    
    if args.single_gpu:
        # Single GPU mode - no DeepSpeed, no DDP
        print("Running in single GPU mode...")
        common_args.update({
            "report_to": "tensorboard",  # Use tensorboard instead of wandb for simplicity
            "deepspeed": None,
        })
    else:
        # Multi-GPU mode with DeepSpeed
        print("Running in multi-GPU mode with DeepSpeed...")
        common_args.update({
            "report_to": "wandb",
            "ddp_find_unused_parameters": False,
        })

    try:
        from trl import DPOConfig  # type: ignore
        training_args: Any = DPOConfig(**common_args)
    except Exception:
        training_args = TrainingArguments(**common_args)

    # Backfill attributes that some TRL versions require on `args`.
    for _name in [
        "model_init_kwargs",
        "ref_model_init_kwargs",
        "accelerator_config",
    ]:
        if not hasattr(training_args, _name):
            setattr(training_args, _name, None)

    print("Initializing PhysioDPO Trainer...")
    try:
        trainer = PhysioDPOTrainer(
            energy_params=PHYSIO_PARAMS,
            model=model,
            ref_model=None, # In LoRA mode, ref_model is None, TRL will automatically handle disable_adapter
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_length=MAX_LENGTH,
            max_prompt_length=128,
        )
        print("Trainer initialized successfully")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        print("\nTrainer initialization failed. Please check:")
        print("TRL version compatibility (try: pip install trl>=0.7.0)")
        raise

    print("Starting Physio-DPO Training...")
    print(f"Training configuration:")
    print(f"  - Model: {MODEL_ID}")
    print(f"  - Dataset: {DATA_PATH} ({len(dataset)} samples)")
    print(f"  - Batch size: {MICRO_BATCH_SIZE} x {GRAD_ACCUM_STEPS} grad accum")
    print(f"  - Max steps: {MAX_STEPS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Max length: {MAX_LENGTH}")
    print(f"  - Physio params: mu={PHYSIO_PARAMS['mu']}, tau={PHYSIO_PARAMS['tau']}, lambda={PHYSIO_PARAMS['lambda']}")
    print(f"  - Single GPU: {args.single_gpu}")
    print()
    
    # Start training with proper callback handling
    trainer.train()
    
    # Ensure training completion callback is called
    if hasattr(trainer, 'on_train_end'):
        trainer.on_train_end()
    
    print("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()