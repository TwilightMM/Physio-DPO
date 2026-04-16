#!/usr/bin/env python3
"""
Convert `preference_dataset.jsonl` into the format required for DPO training.

Input format (JSONL):
{"winner_seq": "...", "loser_seq": "...", "winner_score": 85.5, "loser_score": 23.8, ...}

Output format (JSON):
[{"prompt": "", "chosen": "winner_seq", "rejected": "loser_seq", "energy_gap": delta_score}, ...]
"""

import json
import argparse
from pathlib import Path

def convert_jsonl_to_dpo_format(input_file: str, output_file: str, prompt: str = ""):
    """Convert JSONL data into the DPO training format."""
    samples = []
    
    print(f"Reading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract fields
                winner_seq = data.get('winner_seq', '')
                loser_seq = data.get('loser_seq', '')
                winner_score = data.get('winner_score', 0.0)
                loser_score = data.get('loser_score', 0.0)
                
                if not winner_seq or not loser_seq:
                    print(f"  Warning: line {line_num} is missing winner_seq or loser_seq. Skipping.")
                    continue
                
                # Compute the energy gap (winner_score - loser_score)
                energy_gap = float(winner_score - loser_score)
                
                # Build a DPO-format sample
                sample = {
                    "prompt": prompt,  # Protein generation usually uses an empty prompt
                    "chosen": winner_seq,
                    "rejected": loser_seq,
                    "energy_gap": energy_gap,  # Required by Physio-DPO
                }
                
                # Preserve original metadata when available
                if 'source' in data:
                    sample['source'] = data['source']
                if 'winner_id' in data:
                    sample['winner_id'] = data['winner_id']
                if 'loser_id' in data:
                    sample['loser_id'] = data['loser_id']
                
                samples.append(sample)
                
            except json.JSONDecodeError as e:
                print(f"  Error: failed to parse JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"  Error: failed to process line {line_num}: {e}")
                continue
    
    print(f"Successfully converted {len(samples)} samples")
    
    # Save as a JSON file
    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # Summary statistics
    if samples:
        energy_gaps = [s['energy_gap'] for s in samples]
        print(f"\nDataset statistics:")
        print(f"  - Number of samples: {len(samples)}")
        print(f"  - Average energy gap: {sum(energy_gaps) / len(energy_gaps):.2f}")
        print(f"  - Minimum energy gap: {min(energy_gaps):.2f}")
        print(f"  - Maximum energy gap: {max(energy_gaps):.2f}")
    
    return len(samples)

def main():
    parser = argparse.ArgumentParser(description="Convert preference_dataset.jsonl into DPO training format")
    parser.add_argument(
        "--input", 
        type=str, 
        default="../PhysioDPO_Data/PhysioDPO_Data/output/preference_dataset.jsonl",
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/demo.json",
        help="Path to the output JSON file"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="",
        help="Prompt to prepend to all samples (usually empty for protein generation)"
    )
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check the input file
    if not Path(args.input).exists():
        print(f"Error: input file does not exist: {args.input}")
        print(f"\nHint: run the dataset generation script first to create preference_dataset.jsonl")
        return 1
    
    # Convert the data
    num_samples = convert_jsonl_to_dpo_format(args.input, args.output, args.prompt)
    
    if num_samples > 0:
        print(f"\n✓ Conversion complete. You can start training with:")
        print(f"  bash scripts/train_single_gpu.sh")
        return 0
    else:
        print("\n✗ Conversion failed: no valid samples were found")
        return 1

if __name__ == "__main__":
    exit(main())

