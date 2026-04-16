import random
from typing import List
from core.interfaces import BasePairingStrategy
from core.data_models import ScoredSequence, PreferencePair

class RandomPairer(BasePairingStrategy):
    """
    Randomly pairs stable sequences (winners) with unstable sequences (losers).
    """
    def pair(self, scored_sequences: List[ScoredSequence], **kwargs) -> List[PreferencePair]:
        stable_pool = [s for s in scored_sequences if s.is_stable]
        unstable_pool = [s for s in scored_sequences if not s.is_stable]
        
        pairs = []
        # We can reuse sequences if needed, but for now let's just make unique pairs
        # up to the minimum count
        
        # Shuffle both
        random.shuffle(stable_pool)
        random.shuffle(unstable_pool)
        
        num_pairs = min(len(stable_pool), len(unstable_pool))
        
        for i in range(num_pairs):
            pairs.append(PreferencePair(
                winner=stable_pool[i],
                loser=unstable_pool[i],
                source="random"
            ))
            
        return pairs

class HardNegativePairer(BasePairingStrategy):
    """
    Prioritizes pairing stable sequences with 'hard negative' unstable sequences.
    Without Energy score, we define 'hard negative' as sequences with pLDDT 
    just below the stability threshold (borderline cases).
    """
    def pair(self, scored_sequences: List[ScoredSequence], **kwargs) -> List[PreferencePair]:
        stable_pool = [s for s in scored_sequences if s.is_stable]
        
        # Find hard negatives: Unstable BUT High pLDDT (e.g., > 60 if threshold is 70)
        hard_negatives = []
        for s in scored_sequences:
            if not s.is_stable:
                # Check criteria for "Hard Negative"
                # For pLDDT-only scoring, this means "almost stable"
                if s.structure.mean_plddt > 50.0: # Moderate confidence, but failed strict threshold
                    hard_negatives.append(s)
                    
        pairs = []
        random.shuffle(stable_pool)
        random.shuffle(hard_negatives)
        
        num_pairs = min(len(stable_pool), len(hard_negatives))
        
        for i in range(num_pairs):
            pairs.append(PreferencePair(
                winner=stable_pool[i],
                loser=hard_negatives[i],
                source="hard_negative"
            ))
            
        return pairs

class MutationPairer(BasePairingStrategy):
    """
    Pairs a stable parent sequence with its unstable mutant.
    This provides the strongest signal for DPO: minimal sequence difference, large stability difference.
    """
    def pair(self, scored_sequences: List[ScoredSequence], **kwargs) -> List[PreferencePair]:
        stable_map = {s.sequence.id: s for s in scored_sequences if s.is_stable}
        
        pairs = []
        
        for s in scored_sequences:
            # Look for mutants that are Unstable
            if not s.is_stable and s.sequence.metadata.get("source") == "mutation":
                parent_id = s.sequence.metadata.get("parent_id")
                if parent_id and parent_id in stable_map:
                    winner = stable_map[parent_id]
                    loser = s
                    
                    pairs.append(PreferencePair(
                        winner=winner,
                        loser=loser,
                        source="mutation_contrast"
                    ))
        
        return pairs
