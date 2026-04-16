import argparse
import os
import sys
import logging

# Configure the HuggingFace mirror endpoint (useful when the mirror is faster or more reliable)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from modules.generators import HuggingFaceGenerator
from modules.folders import ESMFoldFolder
from modules.scorers import PhysioScorer
from modules.pairers import RandomPairer, HardNegativePairer, MutationPairer
from pipeline.orchestrator import PhysioDPOPipeline

def main():
    parser = argparse.ArgumentParser(description="Physio-DPO Dataset Construction Pipeline")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of initial sequences to generate (default 10 for demo)")
    # Lower default threshold to 40.0 for demo purposes so we get at least some "stable" sequences
    parser.add_argument("--plddt_threshold", type=float, default=70.0, help="Threshold for stable structure pLDDT")
    parser.add_argument("--mutation_rate", type=float, default=0.05, help="Mutation rate for generating variants")
    parser.add_argument("--n_mutants", type=int, default=2, help="Number of mutants per stable sequence")
    parser.add_argument("--device", type=str, default=None, help="Device to run models on (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    pdbs_dir = os.path.join(args.output_dir, "pdbs")
    
    # 1. Initialize Components
    # Using Real Models: ProtGPT2 for Generation, ESMFold for Folding
    
    # Generator: ProtGPT2 is a good default for protein sequence generation
    generator = HuggingFaceGenerator(
        model_name="nferruz/ProtGPT2", 
        device=args.device,
        min_length=50, 
        max_length=150
    )
    
    # Folder: ESMFold
    folder = ESMFoldFolder(
        model_name="facebook/esmfold_v1",
        device=args.device
    )
    
    scorer = PhysioScorer(plddt_threshold=args.plddt_threshold)
    
    # Pairing strategies
    strategies = [
        RandomPairer(),
        HardNegativePairer(),
        MutationPairer()
    ]
    
    # 2. Initialize Pipeline
    pipeline =PhysioDPOPipeline(
        generator=generator,
        folder=folder,
        scorer=scorer,
        pairing_strategies=strategies
    )
    
    # 3. Run Pipeline Stages
    print(f"--- Starting Physio-DPO Pipeline ---")
    
    # Stage 1: Generate & Fold & Score Initial Batch
    pipeline.run_generation_step(n_samples=args.n_samples, output_dir=args.output_dir)
    
    # Stage 2: Generate Mutants from Stable Sequences (to create close preference pairs)
    pipeline.run_mutation_step(n_mutants_per_stable=args.n_mutants, mutation_rate=args.mutation_rate, output_dir=args.output_dir)
    
    # Stage 3: Create Preference Pairs
    pipeline.create_pairs()
    
    # 4. Save Results
    pipeline.save_results(args.output_dir)

    print(f"--- Pipeline Completed ---")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
