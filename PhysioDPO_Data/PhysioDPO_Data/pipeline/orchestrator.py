from typing import List, Dict, Type
import logging
import os
from core.interfaces import BaseGenerator, BaseFolder, BaseScorer, BasePairingStrategy
from core.data_models import ScoredSequence, PreferencePair
from modules.generators import MutationGenerator
from utils.io_utils import save_scored_sequences, save_preference_pairs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PhysioDPOPipeline:
    """
    Orchestrates the Generate -> Fold -> Score -> Pair workflow.
    """
    def __init__(self, 
                 generator: BaseGenerator,
                 folder: BaseFolder,
                 scorer: BaseScorer,
                 pairing_strategies: List[BasePairingStrategy]):
        self.generator = generator
        self.folder = folder
        self.scorer = scorer
        self.pairing_strategies = pairing_strategies
        self.scored_pool: List[ScoredSequence] = []
        self.preference_dataset: List[PreferencePair] = []

    def run_generation_step(self, n_samples: int, output_dir: str = "./output"):
        logging.info(f"Generating {n_samples} sequences...")
        sequences = self.generator.generate(n_samples)
        logging.info(f"Generated {len(sequences)} sequences.")
        
        # Unload generator to free memory for folding
        if hasattr(self.generator, "unload"):
            self.generator.unload()
        
        logging.info("Folding sequences...")
        structures = self.folder.fold(sequences, output_dir=os.path.join(output_dir, "pdbs"))
        
        logging.info("Scoring sequences...")
        scored = self.scorer.score(sequences, structures)
        self.scored_pool.extend(scored)
        
        n_stable = sum(1 for s in scored if s.is_stable)
        logging.info(f"Added {len(scored)} sequences to pool. Stable: {n_stable}, Unstable: {len(scored) - n_stable}")

    def run_mutation_step(self, n_mutants_per_stable: int = 1, mutation_rate: float = 0.05, output_dir: str = "./output"):
        """
        Takes stable sequences from the current pool, generates mutants, 
        folds and scores them, and adds them to the pool (marked as mutation source).
        """
        logging.info("Running Mutation Step...")
        stable_sequences = [s.sequence for s in self.scored_pool if s.is_stable]
        
        if not stable_sequences:
            logging.warning("No stable sequences found in pool to mutate.")
            return

        logging.info(f"Found {len(stable_sequences)} stable sequences to mutate.")
        
        mutator = MutationGenerator(stable_sequences, n_mutants_per_stable, mutation_rate)
        mutants = mutator.generate()
        
        logging.info(f"Generated {len(mutants)} mutant sequences. Folding...")

        structures = self.folder.fold(mutants, output_dir=os.path.join(output_dir, "pdbs_mutants"))
        
        logging.info("Scoring mutants...")
        scored_mutants = self.scorer.score(mutants, structures)
        
        self.scored_pool.extend(scored_mutants)
        logging.info(f"Added {len(scored_mutants)} mutants to pool.")

    def create_pairs(self):
        logging.info("Creating preference pairs...")
        all_pairs = []
        
        for strategy in self.pairing_strategies:
            pairs = strategy.pair(self.scored_pool)
            logging.info(f"Strategy {strategy.__class__.__name__} generated {len(pairs)} pairs.")
            all_pairs.extend(pairs)
            
        self.preference_dataset = all_pairs
        logging.info(f"Total pairs generated: {len(all_pairs)}")

    def save_results(self, output_dir: str):
        save_scored_sequences(self.scored_pool, f"{output_dir}/scored_sequences.jsonl")
        save_preference_pairs(self.preference_dataset, f"{output_dir}/preference_dataset.jsonl")
        logging.info(f"Results saved to {output_dir}")
