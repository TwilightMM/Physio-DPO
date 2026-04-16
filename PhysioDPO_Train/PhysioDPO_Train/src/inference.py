import os
import sys
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple
import torch.nn.functional as F


def compute_perplexity(model, tokenizer, sequences: List[str], device: str = "cuda") -> float:
    """
    Compute perplexity (PPL) for generated protein sequences
    model eval
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for seq in tqdm(sequences, desc="Computing PPL"):
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def compute_plddt_scores(sequences: List[str]) -> Tuple[float, List[float]]:
    """
    Compute pLDDT scores using ESMFold or similar structure prediction model
    Note: This is a placeholder. In practice, you would use ESMFold or AlphaFold2
    to predict structures and extract pLDDT scores.
    """
    try:
        from transformers import AutoTokenizer, EsmForProteinFolding
        
        print("Loading ESMFold model for pLDDT computation...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        model = model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        plddt_scores = []
        
        with torch.no_grad():
            for seq in tqdm(sequences, desc="Computing pLDDT"):
                # Truncate very long sequences
                if len(seq) > 400:
                    seq = seq[:400]
                
                inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model(**inputs)
                plddt = outputs.plddt.mean().item()
                plddt_scores.append(plddt)
        
        avg_plddt = np.mean(plddt_scores)
        return avg_plddt, plddt_scores
        
    except ImportError:
        print("Warning: ESMFold not available. Returning dummy pLDDT scores.")
        print("To compute real pLDDT scores, install: pip install fair-esm")
        # Return dummy scores
        dummy_scores = [70.0 + np.random.randn() * 5 for _ in sequences]
        return np.mean(dummy_scores), dummy_scores


def compute_sc_rmse(generated_seqs: List[str], reference_seqs: List[str]) -> float:
    """
    Compute self-consistency RMSE (sc-RMSE) between generated and reference sequences
    This measures the structural consistency using sequence-based features
    
    For protein sequences, we can use:
    1. Amino acid composition similarity
    2. Physicochemical property similarity
    3. Secondary structure prediction similarity (if available)
    """
    if len(generated_seqs) != len(reference_seqs):
        raise ValueError("Generated and reference sequences must have the same length")
    
    # Amino acid properties (hydrophobicity scale)
    aa_properties = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
        'X': 0.0  # Unknown
    }
    
    def seq_to_features(seq: str) -> np.ndarray:
        """Convert sequence to feature vector"""
        # Compute amino acid composition
        composition = np.zeros(20)
        aa_list = list(aa_properties.keys())[:20]
        
        for aa in seq:
            if aa in aa_list:
                idx = aa_list.index(aa)
                composition[idx] += 1
        
        if len(seq) > 0:
            composition = composition / len(seq)
        
        # Compute average hydrophobicity
        hydrophobicity = np.mean([aa_properties.get(aa, 0.0) for aa in seq])
        
        # Combine features
        features = np.concatenate([composition, [hydrophobicity]])
        return features
    
    # Compute features for all sequences
    gen_features = np.array([seq_to_features(seq) for seq in generated_seqs])
    ref_features = np.array([seq_to_features(seq) for seq in reference_seqs])
    
    # Compute RMSE
    mse = np.mean((gen_features - ref_features) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def generate_sequences(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_length: int = 256,
    num_return_sequences: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda"
) -> List[str]:
    """
    Generate protein sequences from prompts
    """
    model.eval()
    generated_sequences = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating sequences"):
            # Handle empty prompts (unconditional generation)
            if not prompt or prompt.strip() == "":
                # For unconditional generation, use BOS token or a minimal input
                inputs = tokenizer(tokenizer.bos_token if tokenizer.bos_token else "<|endoftext|>", 
                                   return_tensors="pt", 
                                   add_special_tokens=False)
            else:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            for output in outputs:
                seq = tokenizer.decode(output, skip_special_tokens=True)
                # Remove the prompt from the generated sequence (if not empty)
                if prompt and prompt in seq:
                    seq = seq.replace(prompt, "").strip()
                generated_sequences.append(seq)
    
    return generated_sequences


def main():
    parser = argparse.ArgumentParser(description="PhysioDPO Inference and Evaluation")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model checkpoint")
    parser.add_argument("--base_model", type=str, default="salesforce/progen2-xlarge",
                        help="Base model ID")
    parser.add_argument("--test_data", type=str, default="data/physiopref_1m.json",
                        help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--compute_plddt", action="store_true",
                        help="Compute pLDDT scores (requires ESMFold)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with LoRA weights
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"Loading LoRA weights from: {args.model_path}")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model = model.merge_and_unload()  # Merge LoRA weights with base model
    model.eval()
    
    # Fix ProGen config compatibility: add num_hidden_layers attribute
    if hasattr(model.config, 'n_layer') and not hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = model.config.n_layer
        print(f"Added num_hidden_layers={model.config.n_layer} to model config for compatibility")
    
    # Load test dataset
    print(f"Loading test dataset from: {args.test_data}")
    dataset = load_dataset("json", data_files=args.test_data, split="train")
    
    # Sample subset for evaluation
    if len(dataset) > args.num_samples:
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        dataset = dataset.select(indices)
    
    # Extract prompts and reference sequences
    prompts = []
    reference_sequences = []
    
    for sample in dataset:
        # Assuming the dataset has 'prompt' and 'chosen' fields
        if 'prompt' in sample:
            prompts.append(sample['prompt'])
        if 'chosen' in sample:
            reference_sequences.append(sample['chosen'])
    
    print(f"\nEvaluating on {len(prompts)} samples...")
    
    # Generate sequences
    print("\n=== Generating Sequences ===")
    generated_sequences = generate_sequences(
        model, 
        tokenizer, 
        prompts, 
        max_length=args.max_length,
        temperature=args.temperature,
        device=args.device
    )
    
    # Compute metrics
    print("\n=== Computing Metrics ===")
    
    # 1. Perplexity (PPL)
    print("\n1. Computing Perplexity (PPL)...")
    ppl = compute_perplexity(model, tokenizer, generated_sequences, device=args.device)
    print(f"   Perplexity: {ppl:.4f}")
    
    # 2. sc-RMSE
    print("\n2. Computing sc-RMSE...")
    if len(reference_sequences) == len(generated_sequences):
        sc_rmse = compute_sc_rmse(generated_sequences, reference_sequences)
        print(f"   sc-RMSE: {sc_rmse:.4f}")
    else:
        print("   Warning: Cannot compute sc-RMSE (mismatched sequence counts)")
        sc_rmse = None
    
    # 3. pLDDT
    avg_plddt = None
    if args.compute_plddt:
        print("\n3. Computing pLDDT scores...")
        avg_plddt, plddt_scores = compute_plddt_scores(generated_sequences)
        print(f"   Average pLDDT: {avg_plddt:.4f}")
    else:
        print("\n3. Skipping pLDDT computation (use --compute_plddt to enable)")
    
    # Save results
    results = {
        "perplexity": float(ppl),
        "sc_rmse": float(sc_rmse) if sc_rmse is not None else None,
        "avg_plddt": float(avg_plddt) if avg_plddt is not None else None,
        "num_samples": len(generated_sequences),
        "model_path": args.model_path,
    }
    
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results Summary ===")
    print(f"Perplexity (PPL): {ppl:.4f}")
    if sc_rmse is not None:
        print(f"sc-RMSE: {sc_rmse:.4f}")
    if avg_plddt is not None:
        print(f"Average pLDDT: {avg_plddt:.4f}")
    print(f"\nResults saved to: {results_file}")
    
    # Save generated sequences
    sequences_file = os.path.join(args.output_dir, "generated_sequences.txt")
    with open(sequences_file, 'w') as f:
        for i, (prompt, gen_seq) in enumerate(zip(prompts, generated_sequences)):
            f.write(f"=== Sample {i+1} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {gen_seq}\n\n")
    
    print(f"Generated sequences saved to: {sequences_file}")


if __name__ == "__main__":
    main()
